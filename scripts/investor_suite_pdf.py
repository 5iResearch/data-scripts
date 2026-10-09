"""
Read one Investor Suite article straight from the published PDF, for build_investor_suite.py.

The PDF is what members saw, so for past issues it is the source of truth: the wording is taken from its text
layer and every chart or table is cropped from the page itself (the valuation tables are drawn, not pictures,
so cropping is the only way to get them exactly as published). The Word draft, when there is one, only helps
with structure: which short lines are subheadings and how they are capitalised, where a paragraph starts when
the layout doesn't show it, and which paragraph each chart belongs after.

read_article() returns the same block list read_draft() does (h, p, li, img), so the blog markup is shared.
"""

import html
import itertools
import math
import os
import re
from collections import Counter
from difflib import SequenceMatcher

import fitz
from PIL import Image, ImageOps

LABELS = {"coi": ("CHART OF INTEREST", "CHARTS OF INTEREST"), "musings": ("MARKET MUSINGS",),
          "soi": ("STOCK OF INTEREST",)}
CAPTION = re.compile(r"^(Figure|Chart|Table|Exhibit)\s+\d+\b|^Sources?\s*:", re.I)
# A whole line that is only a caption, for issues that set captions in the article's own font.
CAPTION_LINE = re.compile(r"^(?:(?:Figure|Chart|Table|Exhibit|FIGURE|CHART|TABLE|EXHIBIT)\s+\d+\s*(?:[:.,–—-].{0,80})?"
                          r"|S(?:ources?|OURCES?)\s*:.{0,120})$")
FINE_PRINT = re.compile(r"^\*?\s*(Authors, directors|Employees, directors|Analysts of 5i|Disclosure\s*:|\*\s*As of)",
                        re.I)
THESIS = re.compile(r"investment thesis|(suite|sweet) spot", re.I)
GLYPH = re.compile(r"^[·•▪●◦]\s*")
SYMBOL_FONT = re.compile(r"wingdings|symbol|dingbat", re.I)        # InDesign's bullets: a "y" set in Wingdings
TERMINAL = re.compile(r"[.!?:;…][\"'”’)\]]*$")
PAGE_NO = re.compile(r"^P\.\s*\d+$")
SOFT = "­"
MAX_IMAGE_PX = 1800


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


# --------------------------------------------------------------------------- geometry

def _rect(o):
    return (o["x0"], o["y0"], o["x1"], o["y1"])


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _near(a, b, gap):
    return a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]


def _inside(pt, r):
    return r[0] <= pt[0] <= r[2] and r[1] <= pt[1] <= r[3]


def _centre(o):
    return ((o["x0"] + o["x1"]) / 2, (o["y0"] + o["y1"]) / 2)


def _distance(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def _cluster(rects, gap):
    """Merge rectangles that sit within `gap` points of each other; returns lists of the input indexes."""
    groups = [[i] for i in range(len(rects))]
    boxes = list(rects)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _near(boxes[i], boxes[j], gap):
                    boxes[i] = _union(boxes[i], boxes[j])
                    groups[i] += groups[j]
                    del boxes[j], groups[j]
                    merged = True
                    break
            if merged:
                break
    return groups


# --------------------------------------------------------------------------- page content

def _page_lines(page, n):
    out = []
    for b in page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]:
        if b["type"] != 0:
            continue
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            marker = bool(SYMBOL_FONT.search(spans[0]["font"]))
            kept = [s for s in ln["spans"] if not SYMBOL_FONT.search(s["font"])]
            words = [s for s in kept if s["text"].strip()]
            main = max(words or spans, key=lambda s: len(s["text"].strip()))
            x0, y0, x1, y1 = ln["bbox"]
            out.append(dict(
                page=n, x0=x0, y0=y0, x1=x1, y1=y1, text="".join(s["text"] for s in kept).strip(),
                segs=[(s["text"], bool(s["flags"] & 16) or "bold" in s["font"].lower(),
                       bool(s["flags"] & 2) or "italic" in s["font"].lower()) for s in kept],
                font=main["font"], family=main["font"].split("-")[0], size=round(main["size"], 1),
                color=main["color"], flat=abs(ln["dir"][0]) > 0.9, role="", bullet=marker and bool(words),
                glyph=marker and not words))
    return out


def _repeated(per_page, key, n_pages):
    """Things printed in the same place on many pages: the masthead, page tab, footer."""
    seen = Counter()
    for items in per_page:
        for k in {key(i) for i in items}:
            seen[k] += 1
    return {k for k, c in seen.items() if c >= max(3, n_pages * 0.3)}


def _line_key(l):
    return (re.sub(r"\d+", "#", l["text"]), round(l["x0"] / 8), round(l["y0"] / 8))


def _box_key(r):
    return tuple(round(v / 4) for v in r)


def article_pages(lines_by_page, kind):
    """Pages carrying this article's side tab, leaving out the Stock of Interest cohort table."""
    pages = []
    for i, lines in enumerate(lines_by_page):
        upper = [(" ".join(l["text"].upper().split()), l) for l in lines]
        tab = any(not l["flat"] and any(lbl in t for lbl in LABELS[kind]) for t, l in upper)
        cohort = any("COHORT" in t and l["size"] >= 15 for t, l in upper)
        if tab and not cohort:
            pages.append(i)
    return pages


def _figure_regions(page, lines, images, drawings, body_lines):
    """Rectangles holding a chart or a pasted table, plus the caption groups found near them.

    A region grows from placed pictures and from text set in a font the article's own text doesn't use
    (a table pasted from Excel), then takes in the rules and shading drawn around it."""
    foreign = [l for l in lines if l["role"] == "foreign"]
    seeds, captions = [], []
    for family in {l["family"] for l in foreign}:
        group = [l for l in foreign if l["family"] == family]
        for idx in _cluster([_rect(l) for l in group], 16):
            members = sorted((group[i] for i in idx), key=lambda l: (l["y0"], l["x0"]))
            box = _rect(members[0])
            for m in members[1:]:
                box = _union(box, _rect(m))
            if CAPTION.match(members[0]["text"]):
                captions.append(dict(rect=box, lines=members))
            else:
                seeds.append(box)
    seeds += images
    regions = []
    for idx in _cluster(seeds, 14):
        box = seeds[idx[0]]
        for i in idx[1:]:
            box = _union(box, seeds[i])
        regions.append(box)

    area = page.rect.width * page.rect.height
    body_lines = body_lines + [l for c in captions for l in c["lines"]]
    grown = True
    while grown:
        grown = False
        for d in drawings:
            if (d[2] - d[0]) * (d[3] - d[1]) > area * 0.5:
                continue
            for i, r in enumerate(regions):
                if not _near(r, d, 3) or _union(r, d) == r:
                    continue
                bigger = _union(r, d)
                if any(_inside(_centre(l), bigger) and not _inside(_centre(l), r) for l in body_lines):
                    continue            # a rule that runs on into the article's text isn't part of the chart
                regions[i] = bigger
                grown = True
        for idx in _cluster(regions, 0):
            if len(idx) > 1:
                box = regions[idx[0]]
                for i in idx[1:]:
                    box = _union(box, regions[i])
                regions = [r for i, r in enumerate(regions) if i not in idx] + [box]
                grown = True
                break
    regions = [r for r in regions if r[2] - r[0] >= 60 and r[3] - r[1] >= 30]
    return regions, captions


def _render(page, rect, avoid=(), inner=None):
    """The region as members saw it: cropped from the page, white margins trimmed."""
    clip = fitz.Rect(rect[0] - 3, rect[1] - 3, rect[2] + 3, rect[3] + 3) & page.rect
    if inner:
        clip.x0, clip.x1 = max(clip.x0, inner[0]), min(clip.x1, inner[1])
    for a in avoid:                             # keep the page's side tab out of a chart that sits beside it
        if a[2] - a[0] < 40 and a[1] < clip.y1 and a[3] > clip.y0:
            if clip.x0 < a[2] < rect[0] + 1:
                clip.x0 = a[2] + 0.5
            elif rect[2] - 1 < a[0] < clip.x1:
                clip.x1 = a[0] - 0.5
    zoom = min(4.0, max(2.0, 1600 / max(clip.width, 1)))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    box = im.convert("L").point(lambda v: 255 if v < 245 else 0).getbbox()
    if box:
        pad = 8
        im = im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad),
                      min(im.height, box[3] + pad)))
    if im.width > MAX_IMAGE_PX:
        im = im.resize((MAX_IMAGE_PX, round(im.height * MAX_IMAGE_PX / im.width)), Image.LANCZOS)
    return im


def _reading_order(items, width):
    """Left column top to bottom, then the right; anything spanning both columns splits the page in bands."""
    mid = width / 2

    def column(o):
        if (o["x0"] < mid - 15 and o["x1"] > mid + 60) or o.get("span"):
            return "span"
        return "L" if o["x0"] < mid - 15 else "R"

    for o in items:
        o["col"] = column(o)
    spans = sorted((o for o in items if o["col"] == "span"), key=lambda o: o["y0"])
    out, top = [], -1e9
    for s in spans + [None]:
        bottom = s["y0"] if s else 1e9
        band = [o for o in items if o["col"] != "span" and top <= o["y0"] < bottom]
        out += sorted((o for o in band if o["col"] == "L"), key=lambda o: o["y0"])
        out += sorted((o for o in band if o["col"] == "R"), key=lambda o: o["y0"])
        if s:
            out.append(s)
            top = s["y0"]
    return out


# --------------------------------------------------------------------------- text

def _segments(lines, words, bullet):
    """The lines of one paragraph joined into styled runs, with the layout's hyphenation undone."""
    runs = []
    for n, ln in enumerate(lines):
        segs = [[t.replace(" ", " "), b, i] for t, b, i in ln["segs"]]
        segs[0][0] = segs[0][0].lstrip()
        segs[-1][0] = segs[-1][0].rstrip()
        if n == 0 and bullet:
            segs[0][0] = GLYPH.sub("", segs[0][0])
        if runs:
            last, nxt = runs[-1], "".join(s[0] for s in segs)
            if last[0].endswith(SOFT):
                last[0] = last[0][:-1]
            elif last[0].endswith("-") and re.match(r"\w", nxt):
                a = re.search(r"([\w’']+)-$", "".join(r[0] for r in runs))
                b = re.match(r"[\w’']+", nxt)
                whole = (a.group(1) + b.group(0)).lower() if a and b else ""
                if whole in words and whole.replace(a.group(1).lower(), a.group(1).lower() + "-", 1) not in words:
                    last[0] = last[0][:-1]          # the draft spells it as one word: a line-break hyphen
            else:
                last[0] += " "
        runs += segs
    merged = []
    for t, b, i in runs:
        t = t.replace(SOFT, "")
        if merged and merged[-1][1:] == [b, i]:
            merged[-1][0] += t
        elif t:
            merged.append([t, b, i])
    for m in merged:
        m[0] = re.sub(r"\s{2,}", " ", m[0])
    return merged


def _text(runs):
    return "".join(r[0] for r in runs).strip()


def _html(runs):
    plain = all(r[1] for r in runs) or not any(r[1] or r[2] for r in runs)
    out = ""
    for t, b, i in runs:
        s = html.escape(t, quote=False)
        if not plain:
            s = f"<em>{s}</em>" if i else s
            s = f"<strong>{s}</strong>" if b else s
        out += s
    return out.strip()


def _smart_title(s, caps=()):
    """ALL-CAPS headings in title case, keeping tickers and the acronyms the article itself uses."""
    if s != s.upper():
        return s
    small = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with"}

    def fix(m):
        w = m.group(0)
        if m.start() and w.lower() in small:
            return w.lower()
        if "(" in s[max(0, m.start() - 1):m.start()] or w in caps:
            return w
        return w[0] + w[1:].lower()

    return re.sub(r"[A-Z][A-Z’']*", fix, s)


def _hints(draft_blocks, draft_title):
    starts, headings, words = set(), {}, set()
    if draft_title:
        headings[norm(draft_title)] = draft_title
    for b in draft_blocks or []:
        if b["t"] == "h":
            headings[norm(b["text"])] = b["text"]
        if b["t"] in ("p", "li"):
            if len(norm(b["text"])) >= 18:
                starts.add(norm(b["text"])[:18])
            words.update(re.findall(r"[\w’'-]+", b["text"].lower()))
    return dict(starts=starts, headings=headings, words=words, any=bool(draft_blocks))


# --------------------------------------------------------------------------- figures

def _thumb(im):
    return list(ImageOps.autocontrast(im.convert("L").resize((32, 32), Image.LANCZOS)).getdata())


def _similarity(a, b):
    """-1..1: how alike two pictures look, less a penalty when their shapes differ."""
    ta, tb = _thumb(a), _thumb(b)
    ma, mb = sum(ta) / len(ta), sum(tb) / len(tb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ta, tb))
    va, vb = sum((x - ma) ** 2 for x in ta), sum((y - mb) ** 2 for y in tb)
    corr = cov / math.sqrt(va * vb) if va and vb else 0
    return corr - 0.5 * abs(math.log((a.width / a.height) / (b.width / b.height)))


def _place_figures(blocks, draft_blocks, draft_dir, notes):
    """Move each PDF chart to the spot the draft puts it: after the same paragraph.

    Charts are paired with the draft's by how alike they look; whatever is left over (a chart the draft only
    marks with the word "image") is paired in order. A chart with no partner stays where the PDF has it."""
    figs = [b for b in blocks if b["t"] == "img"]
    slots, anchor = [], None
    for b in draft_blocks or []:
        if b["t"] in ("p", "li", "h"):
            anchor = b
        elif b["t"] in ("img", "gap"):
            slots.append(dict(block=b, anchor=anchor))
    if not figs or not slots:
        return blocks

    def label(caption):
        m = re.search(r"(Figure|Chart|Table|Exhibit)\s+\d+", " ".join(caption), re.I)
        return norm(m.group(0)) if m else ""

    score = {}                                  # (slot, chart) -> how well they go together
    for si, s in enumerate(slots):
        path = os.path.join(draft_dir, s["block"].get("name") or "")
        if s["block"]["t"] != "img" or not os.path.isfile(path):
            continue
        with Image.open(path) as im:
            im.load()
            for fi, f in enumerate(figs):
                look = _similarity(im, f["im"])
                same_label = label(s["block"]["caption"]) and label(s["block"]["caption"]) == label(f["caption"])
                if look >= 0.55:
                    score[si, fi] = look + (0.3 if same_label else 0) - 0.03 * abs(si - fi)
    # The pairing with the best total, so two look-alike tables aren't swapped on a near tie.
    with_picture = sorted({si for si, _ in score})
    pair, best = {}, -1.0
    fewer_slots = len(with_picture) <= len(figs)
    small, large = (with_picture, range(len(figs))) if fewer_slots else (range(len(figs)), with_picture)
    if with_picture and math.perm(len(large), len(small)) <= 200000:
        for combo in itertools.permutations(large, len(small)):
            pairs = zip(small, combo) if fewer_slots else zip(combo, small)
            chosen = {si: fi for si, fi in pairs if (si, fi) in score}
            total = sum(score[k] for k in chosen.items())
            if total > best:
                pair, best = chosen, total
    else:
        for (si, fi), _ in sorted(score.items(), key=lambda kv: -kv[1]):
            if si not in pair and fi not in pair.values():
                pair[si] = fi
    used = set(pair.values())
    by_look = len(pair)
    left = [fi for fi in range(len(figs)) if fi not in used]
    for si in range(len(slots)):
        if si not in pair and left:
            pair[si] = left.pop(0)
    notes.append(f"Chart positions follow the Word draft: {by_look} matched by appearance, "
                 f"{len(pair) - by_look} by order" + (f", {len(left)} left where the PDF has them." if left else "."))

    text_blocks = [b for b in blocks if b["t"] in ("p", "li", "h")]
    keys = [norm(b["text"]) for b in text_blocks]

    def find(anchor):
        if anchor is None:
            return None
        key = norm(anchor["text"])
        probe = key[:60]
        for b, k in zip(text_blocks, keys):
            if probe and (probe in k if anchor["t"] != "h" else k == key):
                return b
        best = max(zip(text_blocks, keys), key=lambda bk: SequenceMatcher(None, key[:120], bk[1][:120]).ratio(),
                   default=None)
        if best and SequenceMatcher(None, key[:120], best[1][:120]).ratio() >= 0.6:
            return best[0]
        return None

    after = {}                                  # id(text block) -> charts to put after it
    moved = set()
    for si in sorted(pair):
        target = find(slots[si]["anchor"])
        if target is not None:
            after.setdefault(id(target), []).append(figs[pair[si]])
            moved.add(id(figs[pair[si]]))
            draft_caption = " ".join(slots[si]["block"].get("caption", []))
            ours = " ".join(figs[pair[si]]["caption"])
            if draft_caption and ours and norm(draft_caption) != norm(ours):
                notes.append(f'Caption kept as published: "{ours}" (the draft has "{draft_caption}").')
    out = []
    for b in blocks:
        if id(b) in moved:
            continue
        out.append(b)
        out += after.get(id(b), [])
    return out


# --------------------------------------------------------------------------- the article

def read_article(pdf_path, kind, draft_blocks=None, draft_title="", draft_dir="", img_dir="", img_stem="",
                 hint_pages=None):
    """dict(title, blocks, thesis, pages, unused, notes) for one article, or None when it can't be found."""
    notes = []
    with fitz.open(pdf_path) as doc:
        lines_by_page = [_page_lines(p, i) for i, p in enumerate(doc)]
        pages = article_pages(lines_by_page, kind)
        if hint_pages and set(pages) <= set(range(min(hint_pages), max(hint_pages) + 1)):
            # Some issues only tab the first page; the draft's own sentences show how far the article runs.
            cohort = {i for i, lines in enumerate(lines_by_page)
                      if any("COHORT" in l["text"].upper() and l["size"] >= 15 for l in lines)}
            pages = [i for i in range(min(hint_pages), max(hint_pages) + 1) if i not in cohort]
        if not pages:
            return None
        n = len(doc)
        fixed_lines = _repeated(lines_by_page, _line_key, n)
        images_by_page = [p.get_image_info(xrefs=True) for p in doc]
        fixed_images = {x for x, c in Counter(i["xref"] for imgs in images_by_page
                                              for i in {v["xref"]: v for v in imgs}.values()).items()
                        if c >= max(3, n * 0.3)}
        drawings_by_page = [[(tuple(d["rect"]), d.get("fill")) for d in p.get_drawings()] for p in doc]
        fixed_boxes = _repeated([[r for r, _ in ds] for ds in drawings_by_page], _box_key, n)

        # What the article's own text is set in.
        weight = Counter()
        for i in pages:
            for l in lines_by_page[i]:
                if l["flat"] and _line_key(l) not in fixed_lines:
                    weight[(l["font"], l["size"])] += len(l["text"])
        if not weight:
            return None
        by_font = Counter()
        for (font, _), chars in weight.items():
            by_font[font] += chars
        body_font = by_font.most_common(1)[0][0]
        body_size = max((chars, size) for (font, size), chars in weight.items() if font == body_font)[1]
        family = body_font.split("-")[0]
        hints = _hints(draft_blocks, draft_title)
        caps = set()

        ordered, unused, blocked = [], [], {}
        for i in pages:
            page, lines = doc[i], lines_by_page[i]
            fills = [r for r, f in drawings_by_page[i] if f and min(f) < 0.94]
            pics = [tuple(v["bbox"]) for v in images_by_page[i]]
            for l in lines:
                white = l["color"] == 0xFFFFFF and not any(_inside(_centre(l), r) for r in fills + pics)
                if l["glyph"]:
                    l["role"] = "glyph"
                elif not l["flat"] or _line_key(l) in fixed_lines or PAGE_NO.match(l["text"]) or white:
                    l["role"] = "skip"
                elif l["size"] >= body_size + 2 and not l["bullet"] and (l["font"] != body_font or len(l["text"]) < 60):
                    l["role"] = "big"
                elif l["family"] != family:
                    l["role"] = "foreign"
                elif FINE_PRINT.match(l["text"]) or (l["font"] != body_font and l["size"] <= body_size - 1.5
                                                     and not l["segs"][0][1]):
                    l["role"] = "fine"
                else:
                    l["role"] = "body"
            # A bullet glyph set as its own little line belongs to the text beside it.
            for g in [l for l in lines if l["role"] == "glyph" or (l["role"] == "body" and l["text"] in ("·", "•"))]:
                mate = min((l for l in lines if l is not g and l["role"] == "body" and abs(l["y0"] - g["y0"]) < 7
                            and 0 <= l["x0"] - g["x1"] < 40), key=lambda l: l["x0"], default=None)
                g["role"] = "skip"
                if mate:
                    mate["bullet"] = True
            # Fine print runs on for a few lines after the one that starts it.
            fine = [l for l in lines if l["role"] == "fine"]
            for l in lines:
                if l["role"] == "body" and l["font"] != body_font and any(
                        f["font"] == l["font"] and 0 < l["y0"] - f["y0"] < 3 * l["size"] for f in fine):
                    l["role"] = "fine"
                    fine.append(l)
            for f in [l for l in lines if l["role"] == "fine" and FINE_PRINT.match(l["text"])]:
                prev = f
                for l in sorted(lines, key=lambda l: l["y0"]):
                    if (l["role"] == "body" and l["font"] == f["font"] and abs(l["x0"] - f["x0"]) < 4
                            and 0 < l["y0"] - prev["y0"] < 1.6 * l["size"]):
                        l["role"], prev = "fine", l
            loose = [l for l in lines if l["role"] == "body" and CAPTION_LINE.match(l["text"])]
            for l in loose:
                l["role"] = "caption"

            body = [l for l in lines if l["role"] in ("body", "big")]
            images = [tuple(v["bbox"]) for v in images_by_page[i]
                      if v["xref"] not in fixed_images and v["bbox"][2] - v["bbox"][0] >= 50
                      and v["bbox"][3] - v["bbox"][1] >= 50
                      and (v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1])
                      < page.rect.width * page.rect.height * 0.85]
            boxes = [r for r, _ in drawings_by_page[i] if _box_key(r) not in fixed_boxes]
            fixed = [r for r, _ in drawings_by_page[i] if _box_key(r) in fixed_boxes]
            regions, captions = _figure_regions(page, lines, images, boxes,
                                                [l for l in lines if l["role"] != "foreign"])
            # The footer strip runs up the right-hand edge over anything placed full width; stop charts short of it.
            # (The left-hand tab is handled in _render: charts often start right beside it.)
            rights = [l["x0"] for l in lines if not l["flat"] and l["x0"] > page.rect.width * 0.9]
            inner = (0, min(rights) - 6 if rights else page.rect.width)

            figures = []
            for r in regions:
                inside = [l for l in lines if l["role"] != "skip" and _inside(_centre(l), r)]
                for l in inside:
                    l["role"] = "figure"
                if "investor considerations" in " ".join(l["text"] for l in inside).lower():
                    continue                    # rebuilt as a box from its figures, not shown as a picture
                figures.append(dict(t="img", x0=r[0], y0=r[1], x1=r[2], y1=r[3], page=i, caption=[],
                                    im=_render(page, r, fixed, inner),
                                    share=min(1.0, (r[2] - r[0]) / max(page.rect.width - 56, 1))))
            for c in captions:
                near = min(figures, key=lambda f: _distance(_rect(f), c["rect"]), default=None)
                if near and (len(figures) == 1 or _distance(_rect(near), c["rect"]) <= 80):
                    parts = []
                    for l in c["lines"]:
                        if CAPTION.match(l["text"]) or not parts:
                            parts.append([])
                        parts[-1].append(l)
                        l["role"] = "caption"
                    for part in parts:
                        text = _text(_segments(part, hints["words"], False))
                        if text not in near["caption"]:
                            near["caption"].append(text)
            unused += [l["text"] for l in lines if l["role"] == "foreign"]
            disclosure, prev = False, None
            for l in sorted((l for l in lines if l["role"] == "fine"), key=lambda l: (l["y0"], l["x0"])):
                if FINE_PRINT.match(l["text"]):
                    disclosure = True
                elif prev is None or l["y0"] - prev["y0"] > 3 * l["size"]:
                    disclosure = False
                if not disclosure:
                    unused.append(l["text"])
                prev = l
            for l in body:
                caps.update(re.findall(r"\b[A-Z][A-Z0-9&]{1,5}\b", l["text"]) if l["role"] == "body" else [])
            for l in loose:
                near = min(figures, key=lambda f: _distance(_rect(f), _rect(l)), default=None)
                if near and l["text"] not in near["caption"]:
                    near["caption"].append(l["text"])
                elif not near:
                    unused.append(l["text"])
            blocked[i] = [_rect(f) for f in figures] + [_rect(l) for l in lines if l["role"] == "caption"]
            items = [l for l in lines if l["role"] in ("body", "big")] + figures
            # A Suite Spot box runs the full width, so its heading splits the page the way its bullets do.
            for h in [l for l in items if l.get("role") == "big" and THESIS.search(l["text"])]:
                h["span"] = any(l.get("role") == "body" and 0 < l["y0"] - h["y0"] < 120
                                and l["x0"] < page.rect.width / 2 - 15 and l["x1"] > page.rect.width / 2 + 60
                                for l in items)
            for o in _reading_order(items, page.rect.width):
                o["page"] = i
                ordered.append(o)

        # Lines into headings, paragraphs and bullets.
        edge = {}
        for o in ordered:
            if o.get("role") == "body":
                k = (o["page"], o["col"])
                edge[k] = max(edge.get(k, 0), o["x1"])
        blocks, cur, waiting, broke = [], None, [], True

        def close():
            nonlocal cur
            if cur:
                blocks.append(cur)
            cur = None
            blocks.extend(waiting)
            waiting.clear()

        for o in ordered:
            if o.get("t") == "img":
                last = cur["lines"][-1] if cur else None
                if cur and cur["t"] != "h" and not TERMINAL.search(last["text"]):
                    waiting.append(o)           # the paragraph carries on past the chart
                else:
                    close()
                    blocks.append(o)
                broke = True
                continue
            if o["role"] == "big":
                same = (cur and cur["t"] == "h" and cur["big"] and cur["lines"][-1]["font"] == o["font"]
                        and cur["lines"][-1]["page"] == o["page"] and 0 < o["y0"] - cur["lines"][-1]["y0"] < 1.7 * o["size"])
                if same:
                    cur["lines"].append(o)
                else:
                    close()
                    cur = dict(t="h", big=True, lines=[o])
                broke = False
                continue
            bullet = o["bullet"] or bool(GLYPH.match(o["text"]))
            last = cur["lines"][-1] if cur else None
            start = len(norm(o["text"])) >= 18 and norm(o["text"])[:18] in hints["starts"]
            if cur is None or cur["t"] == "h" or bullet:
                new = True
            elif not broke and last["page"] == o["page"] and last["col"] == o["col"]:
                gap = o["y0"] - last["y0"]
                around = any(r[1] >= last["y0"] and r[3] <= o["y0"] + 2 and r[0] < o["x1"] and r[2] > o["x0"]
                             for r in blocked[o["page"]])     # a chart or caption the text runs around
                if around and not TERMINAL.search(last["text"]):
                    new = False
                else:
                    new = gap > 1.45 * 1.2 * o["size"] or gap < 0 or (start and bool(TERMINAL.search(last["text"])))
            else:                               # across a column, a page or a chart
                if not TERMINAL.search(last["text"]):
                    new = False
                else:
                    short = last["x1"] < edge.get((last["page"], last["col"]), 0) - 25
                    new = short or start
            if new:
                close()
                cur = dict(t="li" if bullet else "p", big=False, lines=[o])
            else:
                cur["lines"].append(o)
            broke = False
        close()

        # Short lead-in lines that are really subheadings: bold in the PDF, or a heading in the draft.
        out = []
        for b in blocks:
            if b["t"] != "p":
                out.append(b)
                continue
            lines, take = b["lines"], 0
            while take < len(lines) and all(s[1] for s in lines[take]["segs"] if s[0].strip()):
                take += 1
            if take and take < len(lines) and (
                    take > 2 or TERMINAL.search(lines[take - 1]["text"].rstrip(":"))
                    or lines[take - 1]["x1"] > edge.get((lines[take - 1]["page"], lines[take - 1]["col"]), 0) - 20):
                take = 0                        # a bold run-in sentence, not a heading
            if not take:
                joined = ""
                for k, ln in enumerate(lines[:3]):
                    joined += norm(ln["text"].replace(SOFT, ""))
                    if joined in hints["headings"]:
                        take = k + 1
                        break
            if not take and not hints["any"] and len(lines) > 1:
                first, k = lines[0], (lines[0]["page"], lines[0]["col"])
                if (len(first["text"]) <= 70 and not TERMINAL.search(first["text"]) and first["x1"] < edge.get(k, 0) - 40
                        and lines[1]["text"][:1].isupper()):
                    take = 1
            if take and (take < len(lines) or len(_text(_segments(lines, set(), False))) <= 140):
                out.append(dict(t="h", big=False, lines=lines[:take]))
                if lines[take:]:
                    out.append(dict(t="p", big=False, lines=lines[take:]))
            else:
                out.append(b)

        # Finished blocks.
        title, final, thesis, section = "", [], {}, None
        labels = {norm(x) for v in LABELS.values() for x in v}
        has_big = False
        for b in out:
            if b["t"] == "img":
                final.append(b)
                continue
            runs = _segments(b["lines"], hints["words"], b["t"] == "li")
            text = _text(runs)
            if not text:
                continue
            if b["t"] == "h":
                if norm(text) in labels:
                    continue
                shown = hints["headings"].get(norm(text)) or _smart_title(text, caps)
                if b["big"] and not title:
                    title = re.sub(r"^(%s)\s*[-–—:]\s*" % "|".join(x for v in LABELS.values() for x in v), "",
                                   shown, flags=re.I)
                    continue
                if norm(shown) == norm(title):
                    continue                    # the title printed a second time
                if THESIS.search(text):
                    section = thesis.setdefault(shown, [])
                    continue
                section = None
                has_big |= b["big"]
                final.append(dict(t="h", text=shown, big=b["big"]))
            elif section is not None and (b["t"] == "li" or len(text) <= 260):
                section.append(text)
            elif FINE_PRINT.match(text):
                continue
            elif CAPTION.match(text) and len(text) <= 200:
                first = b["lines"][0]
                figs = [f for f in final if f["t"] == "img" and f["page"] == first["page"]] or \
                       [f for f in out if f["t"] == "img" and f["page"] == first["page"]]
                near = min(figs, key=lambda f: _distance(_rect(f), _rect(first)), default=None)
                if near and text not in near["caption"]:
                    near["caption"].append(text)
                elif not near:
                    unused.append(text)
            else:
                section = None
                final.append(dict(t=b["t"], text=text, html=_html(runs)))
        for b in final:
            if b["t"] == "h":
                b["level"] = 1 if b.pop("big") or not has_big else 2

        final = _place_figures(final, draft_blocks, draft_dir, notes)
        n_fig = 0
        for b in final:
            if b["t"] != "img":
                continue
            n_fig += 1
            im = b.pop("im")
            b["name"] = f"{img_stem}-fig{n_fig}.png"
            im.save(os.path.join(img_dir, b["name"]), "PNG", optimize=True)
            b["w"], b["h"] = im.size
            for k in ("x0", "y0", "x1", "y1", "col"):
                b.pop(k, None)
        return dict(title=title, blocks=final, thesis=thesis, pages=pages, unused=unused, notes=notes)
