from itertools import chain
import sys

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode


def get_supported_glyphs(font_path):
    ttf = TTFont(font_path, 0, allowVID=0,
                 ignoreDecompileErrors=True,
                 fontNumber=-1)

    chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)
    chars = list(set(chars))
    chars.sort(key=lambda x: x[0])
    names = [c[2] for c in chars]
    chars = [chr(c[0]) for c in chars]
    ttf.close()

    return chars, names


def main():
    chars, _ = get_supported_glyphs(sys.argv[1])
    print(' '.join(chars))
    print(len(chars))


if __name__ == '__main__':
    main()

