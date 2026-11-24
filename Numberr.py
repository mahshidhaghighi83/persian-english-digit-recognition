
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MODEL_PATH = "digit_model.pt"
FA = "۰۱۲۳۴۵۶۷۸۹"
# کلاس‌ها: ۰..۹ = انگلیسی، ۱۰..۱۹ = فارسی، ۲۰ = «عدد نیست» (مثلاً حرف روی پلاک)
LABELS = [f"en{i}" for i in range(10)] + [f"fa{i}" for i in range(10)] + ["other"]
OTHER = 20
REMOVE_LINES = False  # اگر True شود، خط‌های افقی بلند قبل از پیدا کردن رقم‌ها پاک می‌شوند
LETTERS = "بپتثجچحخدذرزژسشصضطظعغفقکگلمنی" + "ACDEFHJKLMNPRTUVWXY"
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ----------------------------------------------------------------------------
# پیش‌پردازش یک رقم: سفید روی سیاه، برش تنگ، تبدیل به 28x28 (مثل MNIST)
# ----------------------------------------------------------------------------
def normalize_digit(b):
    ys, xs = np.where(b > 0)
    if len(xs) == 0:
        return None
    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = b.shape
    s = 20.0 / max(h, w)
    b = cv2.resize(b, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.uint8)
    h, w = b.shape
    y0, x0 = (28 - h) // 2, (28 - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = b
    return canvas


def to_binary(gray):
    """عکس خاکستری یک رقم -> سفید(جوهر) روی سیاه، با تشخیص خودکار زمینه."""
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if border.mean() > 127:
        gray = 255 - gray
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def imread_unicode(path, flag=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)  # مسیرهای فارسی در ویندوز هم کار کند
    img = cv2.imdecode(data, flag)
    if img is None:
        sys.exit(f"cannot read image: {path}")
    return img


# ----------------------------------------------------------------------------
# مدل: یک CNN کوچک
# ----------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 3 * 3, len(LABELS)))

    def forward(self, x):
        return self.head(self.f(x))


# ----------------------------------------------------------------------------
# داده‌ی آموزش
# ----------------------------------------------------------------------------
def augment(img):
    ang = random.uniform(-12, 12)
    sc = random.uniform(0.85, 1.1)
    shear = random.uniform(-0.2, 0.2)
    M = cv2.getRotationMatrix2D((14, 14), ang, sc)
    M[0, 1] += shear
    M[0, 2] += -shear * 14 + random.uniform(-2, 2)
    M[1, 2] += random.uniform(-2, 2)
    out = cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_LINEAR)
    r = random.random()
    if r < 0.2:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))   # خط ضخیم‌تر
    elif r < 0.4:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))    # خط نازک‌تر
    if random.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


class DigitSet(Dataset):
    def __init__(self, X, y, do_augment):
        self.X, self.y, self.do_augment = X, y, do_augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.do_augment:
            img = augment(img)
        t = torch.from_numpy(np.ascontiguousarray(img)).float().div(255).unsqueeze(0)
        return t, int(self.y[i])


def load_folder(root, name_to_class, limit=None):
    X, y = [], []
    for name, cls in name_to_class.items():
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        files = [f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(IMG_EXT)]
        random.shuffle(files)
        for f in files[:limit]:
            try:
                n = normalize_digit(to_binary(imread_unicode(f, cv2.IMREAD_GRAYSCALE)))
            except Exception:
                continue
            if n is not None:
                X.append(n)
                y.append(cls)
    return X, y


def load_mnist(per_class):
    from torchvision import datasets
    ds = datasets.MNIST("data/mnist", train=True, download=True)
    data, targets = ds.data.numpy(), ds.targets.numpy()
    X, y = [], []
    for d in range(10):
        idx = np.where(targets == d)[0]
        np.random.shuffle(idx)
        for i in idx[:per_class]:
            n = normalize_digit(((data[i] > 80) * 255).astype(np.uint8))
            if n is not None:
                X.append(n)
                y.append(d)
    return X, y


def synth_printed(fonts_dir, per_class):
    """ارقام و حروف چاپی را با فونت‌های شما می‌سازد."""
    from PIL import Image, ImageDraw, ImageFont
    paths = glob.glob(os.path.join(fonts_dir, "*.ttf")) + glob.glob(os.path.join(fonts_dir, "*.otf"))
    if not paths:
        print("WARNING: no fonts found -> printed digits will NOT be learned well.")
        return [], []

    def render(font, ch):
        img = Image.new("L", (128, 128), 0)
        ImageDraw.Draw(img).text((64, 64), ch, font=font, fill=255, anchor="mm")
        return np.array(img)

    fonts = []
    for p in paths:
        try:
            f = ImageFont.truetype(p, 80)
            fonts.append((f, render(f, "\U0010FFFF")))  # شکل «گلیف گمشده» برای فیلتر کردن
        except Exception:
            pass

    def sample(ch):
        for _ in range(10):
            f, tofu = random.choice(fonts)
            arr = render(f, ch)
            if arr.max() == 0 or np.array_equal(arr, tofu):
                continue  # این فونت این نویسه را ندارد
            return normalize_digit(((arr > 100) * 255).astype(np.uint8))
        return None

    X, y = [], []
    for cls in range(20):
        ch = str(cls) if cls < 10 else FA[cls - 10]
        for _ in range(per_class):
            n = sample(ch)
            if n is not None:
                X.append(n)
                y.append(cls)
    for _ in range(per_class * 2):
        n = sample(random.choice(LETTERS))
        if n is not None:
            X.append(n)
            y.append(OTHER)
    return X, y


def train(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    Xs, ys = [], []

    print("loading English handwritten (MNIST)...")
    X, y = load_mnist(args.per_class)
    Xs += X
    ys += y

    if args.persian_dir:
        print("loading Persian handwritten...")
        X, y = load_folder(args.persian_dir, {**{str(d): 10 + d for d in range(10)}, "4.1": 14, "6.1": 16}, args.per_class)
        Xs += X
        ys += y
    else:
        print("WARNING: --persian-dir not given -> Persian handwriting will not work.")

    if args.fonts_dir:
        print("rendering printed digits from fonts...")
        X, y = synth_printed(args.fonts_dir, args.synth_per_class)
        Xs += X
        ys += y

    if args.extra_dir:
        print("loading your own samples...")
        X, y = load_folder(args.extra_dir, {n: i for i, n in enumerate(LABELS)})
        Xs += X * args.extra_weight
        ys += y * args.extra_weight
        print(f"  {len(X)} samples (repeated x{args.extra_weight})")

    X, y = np.stack(Xs), np.array(ys)
    print("samples per class:", {LABELS[i]: int((y == i).sum()) for i in range(len(LABELS))})

    perm = np.random.permutation(len(X))
    n_val = max(1, int(0.1 * len(X)))
    va, tr = perm[:n_val], perm[n_val:]
    tr_loader = DataLoader(DigitSet(X[tr], y[tr], True), batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(DigitSet(X[va], y[va], False), batch_size=256)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.epochs * len(tr_loader))
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
            sched.step()
        model.eval()
        ok = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                ok += (model(xb.to(dev)).argmax(1).cpu() == yb).sum().item()
        acc = ok / len(va)
        print(f"epoch {ep}/{args.epochs}  validation accuracy: {acc:.4f}")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved best model to {MODEL_PATH} (acc {best:.4f})")


# ----------------------------------------------------------------------------
# پیدا کردن ارقام در عکس
# ----------------------------------------------------------------------------
def remove_long_lines(bw):
    """خط‌های افقی بلند (مثل خط زیر نوشته یا لبهٔ کاغذ) را پاک می‌کند."""
    H, W = bw.shape
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, W // 4), 1))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    return cv2.subtract(bw, lines)


def prepare(img, ink="auto"):
    h, w = img.shape[:2]
    s = min(1000.0 / max(h, w), 4.0)
    if abs(s - 1) > 1e-3:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if ink == "auto":
        bright_ink = (otsu > 0).mean() < 0.5  # جوهر معمولاً اقلیت پیکسل‌هاست
    else:
        bright_ink = ink == "bright"
    H, W = gray.shape
    block = max(31, int(max(H, W) / 15) | 1)
    mode = cv2.THRESH_BINARY if bright_ink else cv2.THRESH_BINARY_INV
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, block, 10)
    if REMOVE_LINES:
        bw = remove_long_lines(bw)
    return img, bw


def merge_boxes(boxes):
    """تکه‌های یک رقم (مثلاً رقمی که با دو خط نوشته شده) را یکی می‌کند."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            for b in boxes:
                ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                gy = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
                new_h = max(a[1] + a[3], b[1] + b[3]) - min(a[1], b[1])
                if ox > 0.5 * min(a[2], b[2]) and gy < 0.3 * max(a[3], b[3]) and new_h <= 1.5 * max(a[3], b[3]):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1 = max(a[0] + a[2], b[0] + b[2])
                    y1 = max(a[1] + a[3], b[1] + b[3])
                    boxes.remove(b)
                    a = [x0, y0, x1 - x0, y1 - y0]
                    changed = True
                    break
            out.append(a)
        boxes = out
    return [tuple(b) for b in boxes]


def find_boxes(bw):
    H, W = bw.shape
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_boxes = [cv2.boundingRect(c) for c in cnts]
    all_boxes = [b for b in all_boxes if b[2] * b[3] >= 20]
    boxes = [b for b in all_boxes if b[3] <= 0.8 * H and b[2] <= 0.8 * W]  # قاب پلاک و ... حذف
    if not boxes:
        boxes = all_boxes  # مثلاً عکسی که فقط یک رقم درشت دارد
    if not boxes:
        return []
    boxes = merge_boxes(boxes)
    ref_h = np.percentile([b[3] for b in boxes], 75)
    boxes = [b for b in boxes if 0.2 * ref_h <= b[3] <= 2.0 * ref_h and b[2] <= 2.5 * b[3]
             and b[2] >= 0.04 * b[3]]  # خط عمودی بسیار باریک (لبهٔ کاغذ) رقم نیست
    # تکه‌های ریز چسبیده به لبهٔ عکس معمولاً رقم نیستند (سایه، لبهٔ کاغذ)
    boxes = [b for b in boxes
             if not ((b[0] <= 3 or b[1] <= 3 or b[0] + b[2] >= W - 3 or b[1] + b[3] >= H - 3)
                     and b[3] < 0.5 * ref_h)]
    return boxes


def group_lines(boxes):
    ref = np.median([b[3] for b in boxes])
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
    lines, cur = [], [boxes[0]]
    for b in boxes[1:]:
        cy = np.mean([c[1] + c[3] / 2 for c in cur])
        if abs(b[1] + b[3] / 2 - cy) > 0.6 * ref:
            lines.append(cur)
            cur = [b]
        else:
            cur.append(b)
    lines.append(cur)
    return [sorted(l, key=lambda b: b[0]) for l in lines]


# ----------------------------------------------------------------------------
# خواندن عکس
# ----------------------------------------------------------------------------
def read_image(path, args):
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    img, bw = prepare(imread_unicode(path), args.ink)
    boxes = find_boxes(bw)
    if not boxes:
        print("no digits found")
        return []
    ref_h = float(np.median([b[3] for b in boxes]))
    annotated = img.copy()
    results = []

    for line in group_lines(boxes):
        crops = [normalize_digit(bw[y:y + h, x:x + w]) for (x, y, w, h) in line]
        keep = [i for i, c in enumerate(crops) if c is not None]
        line = [line[i] for i in keep]
        crops = [crops[i] for i in keep]
        if not line:
            continue
        x = torch.from_numpy(np.stack(crops)).float().div(255).unsqueeze(1)
        with torch.no_grad():
            P = F.softmax(model(x), 1).numpy()
        is_other = P.argmax(1) == OTHER

        # شماره‌ی «کلمه» برای هر رقم (فاصله‌ی زیاد یا حرف = کلمه‌ی جدید)
        word_id, wid = [], 0
        for i, (bx, by, bw_, bh) in enumerate(line):
            if i > 0:
                gap = bx - (line[i - 1][0] + line[i - 1][2])
                if gap > args.space_ratio * ref_h or is_other[i] or is_other[i - 1]:
                    wid += 1
            word_id.append(wid)

        # تصمیم‌گیری درباره‌ی نوع رقم‌ها (فارسی/انگلیسی)
        # پیش‌فرض: همه‌ی ارقام یک خط هم‌نوع‌اند (با --mixed: هر کلمه جدا)
        groups = {}
        for i in range(len(line)):
            if not is_other[i]:
                groups.setdefault(word_id[i] if args.mixed else 0, []).append(i)
        script, value = {}, {}
        for idx in groups.values():
            if len(idx) >= 2:
                s = 0 if P[idx, :10].sum() >= P[idx, 10:20].sum() else 1
                for i in idx:
                    script[i] = s
                    value[i] = int(P[i, s * 10:(s + 1) * 10].argmax())
            else:
                i = idx[0]
                c = int(P[i, :20].argmax())
                script[i], value[i] = c // 10, c % 10

        text, norm = [], []
        for i, (bx, by, bw_, bh) in enumerate(line):
            if i > 0 and word_id[i] != word_id[i - 1]:
                text.append(" ")
                norm.append(" ")
            if is_other[i]:
                text.append("-")
                norm.append("-")
                color, label = (255, 0, 0), "-"
            else:
                v = value[i]
                conf = P[i, v] + P[i, 10 + v]  # احتمال «این مقدار»، فارسی یا انگلیسی
                if conf < args.min_conf:
                    text.append("?")
                    norm.append("?")
                    color, label = (0, 0, 255), "?"
                else:
                    text.append(FA[v] if script[i] == 1 else str(v))
                    norm.append(str(v))
                    color, label = (0, 160, 0), f"{v}{'F' if script[i] else 'E'}"
            cv2.rectangle(annotated, (bx, by), (bx + bw_, by + bh), color, 2)
            cv2.putText(annotated, label, (bx, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        results.append(("".join(text), "".join(norm)))

    out_path = args.out or os.path.splitext(path)[0] + "_out.png"
    ok, buf = cv2.imencode(".png", annotated)
    if ok:
        buf.tofile(out_path)
    print(f"annotated image: {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Persian + English digit recognition")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--persian-dir", help="folder with subfolders 0..9 of Persian handwritten digits")
    t.add_argument("--fonts-dir", help="folder of .ttf/.otf fonts for printed digits")
    t.add_argument("--extra-dir", help="your own samples: subfolders en0..en9, fa0..fa9, other")
    t.add_argument("--extra-weight", type=int, default=20, help="repeat your own samples this many times")
    t.add_argument("--per-class", type=int, default=3000)
    t.add_argument("--synth-per-class", type=int, default=2000)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch", type=int, default=128)

    r = sub.add_parser("read")
    r.add_argument("image")
    r.add_argument("--min-conf", type=float, default=0.5, help="below this a digit is shown as ?")
    r.add_argument("--space-ratio", type=float, default=0.8, help="gap (x digit height) that starts a new word")
    r.add_argument("--ink", choices=["auto", "dark", "bright"], default="auto")
    r.add_argument("--mixed", action="store_true", help="allow different scripts in one line")
    r.add_argument("--out", help="path for the annotated output image")

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        if not os.path.exists(MODEL_PATH):
            sys.exit("model not found. run:  python digits.py train ...")
        for text, norm in read_image(args.image, args):
            print(text if text == norm else f"{text}   (= {norm})")


if __name__ == "__main__":
    main()