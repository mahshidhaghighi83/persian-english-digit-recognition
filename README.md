# Persian + English Digit Recognition / تشخیص اعداد فارسی و انگلیسی

A small computer-vision project in Python that reads digits from an image.
It works with **Persian (۰–۹) and English (0–9) digits**, **printed or handwritten**
(for example a license plate or a handwritten note).

یک پروژهٔ کوچک بینایی ماشین با پایتون که اعداد داخل یک عکس را می‌خواند.
اعداد **فارسی و انگلیسی**، **چاپی یا دست‌نویس** را پشتیبانی می‌کند.

## How it works / روش کار

1. The image is converted to black and white and each digit is cut out
   (connected-component boxes, grouped into lines).
2. Each digit is resized to 28×28 and classified by a small CNN (PyTorch).
3. The CNN has 21 classes: 10 English digits, 10 Persian digits, and `other`
   (letters, e.g. on a plate).
4. Persian and English digits are kept as separate classes because some look alike
   (Persian ۵ ≈ English 0, Persian ۶ ≈ English 9). If all digits in a line are of one
   script, the program uses that to fix doubtful digits.

## Files / فایل‌ها

| File | Purpose |
|---|---|
| `Numberr.py` | Training and reading (command line) |
| `gui.py` | Simple graphical window (choose an image, see the result) |
| `collect_fonts.py` | Collects fonts from a folder for generating printed digits |
| `requirements.txt` | Python packages |

## Install / نصب

```
pip install -r requirements.txt
```

## Data and fonts (not included) / داده و فونت (در مخزن نیست)

Training data is **not** in this repository because of licenses:

- **English handwritten:** MNIST, downloaded automatically by the script.
- **Persian handwritten:** download a dataset such as
  [Persian Handwritten Digits and symbols Dataset (Mendeley, CC BY-NC-ND 4.0)](https://data.mendeley.com/datasets/8993swvpjx)
  and put the digit folders `0` … `9` (and `4.1`, `6.1` if present) in `data/persian/train`.
  Please respect its license (non-commercial, no redistribution).
- **Printed digits:** generated from `.ttf` fonts you put in the `fonts` folder
  (for example from [Google Fonts](https://fonts.google.com)). You can run
  `python collect_fonts.py "path/to/downloaded/fonts"` to copy suitable fonts.
- **Optional, your own handwriting:** folders `extra/en0 … en9`, `extra/fa0 … fa9`, `extra/other`
  with cropped digit images, used with `--extra-dir extra`.

## Usage / استفاده

Train (once):

```
python Numberr.py train --persian-dir data/persian/train --fonts-dir fonts --epochs 100
```

Read an image:

```
python Numberr.py read image.jpg
```

Graphical window:

```
python gui.py
```

`gui.py` needs `Numberr.py` and the trained `digit_model.pt` in the same folder.

## Limitations / محدودیت‌ها

- Works best on **clear, separated digits** (printed text, plates, neat handwriting).
- **Touching or overlapping digits** are often not separated correctly, and very thick
  brush-style handwriting is read poorly.
- The program reads **digits only**, not words or sentences.
- Accuracy on the validation set during training was about 99%, but this is measured
  on the same kind of data used for training, not on arbitrary photos.

## License and credits

- Dataset credits: MNIST; Persian Handwritten Digits and symbols Dataset (Mendeley Data).
- Fonts are subject to their own licenses.
- Add your own license for the code here (for example MIT) if you want others to reuse it.
