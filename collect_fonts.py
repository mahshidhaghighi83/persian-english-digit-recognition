
import os
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont

DEST = "fonts"
MAX_PER_FAMILY = 3
EXTS = (".ttf", ".otf")
FA = "۰۱۲۳۴۵۶۷۸۹"
EN = "0123456789"
SKIP_WORDS = ("italic", "oblique", "condensed", "thin", "light", "black",
              "extra", "semi", "demi", "hairline", "heavy")
KEEP_WORDS = ("regular", "medium", "bold", "variablefont")


def render(font, ch):
    img = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(img).text((64, 64), ch, font=font, fill=255, anchor="mm")
    return img.tobytes()


def coverage(path):
    """برمی‌گرداند (انگلیسی کامل است؟, فارسی کامل است؟) یا None اگر فونت باز نشد."""
    try:
        font = ImageFont.truetype(path, 80)
        tofu = render(font, "\U0010FFFF")  # شکل «گلیف گمشده»
        def ok(chars):
            for c in chars:
                data = render(font, c)
                if data == tofu or not any(data):
                    return False
            return True
        return ok(EN), ok(FA)
    except Exception:
        return None


def family_of(stem):
    for sep in ("-", "_", "["):
        stem = stem.split(sep)[0]
    return stem.lower()


def wanted(stem):
    s = stem.lower()
    if any(w in s for w in SKIP_WORDS):
        return False
    return any(w in s for w in KEEP_WORDS) or "-" not in s and "_" not in s


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "Downloads")
    if not os.path.isdir(src):
        sys.exit(f"folder not found: {src}")
    os.makedirs(DEST, exist_ok=True)
    dest_abs = os.path.abspath(DEST)

    found = []
    for root, dirs, files in os.walk(src):
        if os.path.abspath(root).startswith(dest_abs):
            continue
        for f in files:
            if f.lower().endswith(EXTS):
                found.append(os.path.join(root, f))
    print(f"{len(found)} font files found in {src}")

    per_family = {}
    copied = skipped = 0
    for path in sorted(found):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        if not wanted(stem):
            skipped += 1
            continue
        fam = family_of(stem)
        if per_family.get(fam, 0) >= MAX_PER_FAMILY:
            skipped += 1
            continue
        cov = coverage(path)
        if cov is None:
            print(f"  [cannot open] {name}")
            continue
        en, fa = cov
        if not (en or fa):
            print(f"  [no digits]   {name}")
            continue
        target = os.path.join(DEST, name)
        if os.path.exists(target):
            print(f"  [already]     {name}")
            per_family[fam] = per_family.get(fam, 0) + 1
            continue
        shutil.copyfile(path, target)
        per_family[fam] = per_family.get(fam, 0) + 1
        copied += 1
        print(f"  [copied] {name:50s} EN:{'yes' if en else 'no '}  FA:{'yes' if fa else 'no '}")

    print(f"\ncopied {copied} fonts, skipped {skipped}")
    allfonts = [f for f in os.listdir(DEST) if f.lower().endswith(EXTS)]
    n_fa = sum(1 for f in allfonts if (coverage(os.path.join(DEST, f)) or (0, 0))[1])
    print(f"total in '{DEST}': {len(allfonts)} fonts, {n_fa} of them have Persian digits")
    if n_fa < 5:
        print("WARNING: fewer than 5 fonts with Persian digits - add more Persian fonts.")


if __name__ == "__main__":
    main()