"""
Bi-weekly member Market Update -> Mailchimp email + blog post + copy/paste kit.

Runs locally (not in Actions) against an issue folder such as
  Documents/Mailchimp/2026/Sep/Sep 15/
which holds:
  issue.md                  subject, preview, report updates, coverage drops,
                            market intro, model portfolio trades, call to action
                            (a starter template is written if it's missing)
  Market Update - *.docx    the market update, written as usual in Word
  Report Card.png           the Macro Economic Report Card export

Writes <issue>/_build/ (kit.html, email.html, post.html, images) and copies
only the images to outputs/market-update-images/<date>/ in this repo, since
Mailchimp and the blog need them hosted (GitHub Pages). The text - including
model portfolio trades - never leaves the laptop.

Usage:  python scripts/build_market_update.py "<issue folder>"
        (or drag the folder onto "Build Market Update.bat")
Needs python-docx (in the miniconda base env; not in requirements.txt).
"""

import glob
import html
import json
import os
import re
import shutil
import sys
import webbrowser
from datetime import datetime

import docx
from docx.oxml.ns import qn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_TEMPLATE = os.path.join(REPO_ROOT, "templates", "market_update_kit.html")
IMAGES_ROOT = os.path.join(REPO_ROOT, "outputs", "market-update-images")
PAGES_BASE = "https://5iresearch.github.io/data-scripts/outputs/market-update-images"

LOGO_URL = ("https://mcusercontent.com/70a01e3dae2f947876a36d9c2/images/"
            "5d9b0651-152b-36a5-671e-f1157fa1735f.png")
REPORTS_URL = "https://www.5iresearch.ca/reports"
UPGRADE_URL = "https://www.5iresearch.ca/my-account/subscription"

# 5i brand colours
BLUE = "#1F79BE"
ORANGE = "#C67A29"
INK = "#363636"
DBLUE = "#4B8EA9"
GREEN = "#44A660"
RED = "#A22A2A"
MUTED = "#6F6F6F"
RULE = "#E3E6EA"
PANEL = "#F6F8FA"
BLUE_TINT = "#EEF5FB"
ORANGE_TINT = "#FCF3EA"
FONT = "'Helvetica Neue',Helvetica,Arial,sans-serif"

PORTFOLIO_COLOURS = {"growth": BLUE, "balanced": DBLUE, "income": GREEN}
TRADE_BADGES = [  # (first-word prefixes, label, colour)
    (("sell", "exit"), "SELL", RED),
    (("trim", "reduce"), "TRIM", ORANGE),
    (("initiate", "buy", "new"), "BUY", GREEN),
    (("increase", "add", "top"), "ADD", BLUE),
    (("swap", "switch", "replace"), "SWAP", DBLUE),
]

SOCIAL = [
    ("Facebook", "https://www.facebook.com/5iresearch", "facebook"),
    ("Instagram", "https://www.instagram.com/5iresearch/", "instagram"),
    ("X", "https://twitter.com/5iresearchdotca", "twitter"),
    ("YouTube", "https://www.youtube.com/5iresearchinc", "youtube"),
    ("LinkedIn", "https://www.linkedin.com/company/5i-research-inc./", "linkedin"),
    ("TikTok", "https://www.tiktok.com/@5i.research", "tiktok"),
]
SOCIAL_ICON = "https://cdn-images.mailchimp.com/icons/social-block-v3/block-icons-v3/{}-filled-dark-40.png"

PORTFOLIO_ANALYTICS = [
    ("Track your investments across geography, asset class &amp; sector",
     "View all your holdings with detailed breakdowns to spot trends quickly."),
    ("Proprietary macro market models",
     "Use our in-house models for insight on market direction, sector rotation and risk indicators."),
    ("The best U.S. stock picks",
     "Curated monthly U.S. stock recommendations and actionable ideas, exclusive to Portfolio Analytics members."),
    ("Interactive tools &amp; visuals",
     "Company news alerts for your portfolio and watchlist, plus sector and index heatmaps for Canada and the U.S."),
]

DISCLAIMER = (
    "5i Research (5i) is not a registered investment advisor. Any information, recommendations or statements "
    "of opinion provided here and throughout the 5i website are for general information purposes only. It is not "
    "intended to be personalized investment advice or a solicitation for the purchase or sale of securities. The "
    "information contained in this publication are obtained from, or based upon publicly available sources that we "
    "believe to be reliable. 5i makes no warranty as to their accuracy or usefulness of the information provided. "
    "Past performance is not indicative of future results. You further agree that 5i Research will not be liable for "
    "any losses or liabilities that may be occasioned as a result of the information or commentary provided. Do not "
    "buy or sell any stock without conducting your own due diligence or consulting an advisor. Opinions and views "
    "expressed throughout the 5i websites may change and/or differ from the opinions of individuals employed by 5i "
    "Research and/or affiliated companies. Employees of 5i Research involved in the research process cannot trade in "
    "Canadian traded stocks. Employees, directors, officers, related companies, and/or partners may hold a financial "
    "or other interest in funds or US and international securities mentioned. 5i Research is affiliated through "
    "common ownership with i2i Capital Management. Employees, directors, officers, and/or partners hold a financial "
    "or other interest in the i2i Long/Short US Equity Fund (i2i Fund) which from time to time may hold a financial "
    "or other interest in non-Canadian securities discussed throughout the 5i website. Clients of i2i Capital "
    "Managements Private Investment Counsel service (i2i PIC) may hold a financial interest in any companies "
    "discussed and the views of i2i PIC may differ from the views of 5i Research. All information on this website "
    "is intended for Canadian residents only."
)

ISSUE_TEMPLATE = """Subject: Market, Model Portfolio, and Report Updates!
Preview: One line that shows next to the subject in the inbox
Blog title:
Date:

## Report Updates
We have posted report updates on the following names:

- **Aritzia (ATZ):** one-line description.
- **Example (EX):** one-line description.

Optional closing line, e.g. "We have downgraded one of the names."

## New Coverage
Delete this section if there's no new coverage.

## Dropping Coverage
We are dropping coverage on the following names:

- **Example (EX):** reason.

Delete this section if nothing was dropped.

## Market Intro
A short intro paragraph that leads into the market update from the Word doc.

## Model Portfolio: Growth
### Sell Full Example (EX) Position
Trade rationale goes here.

### Increase Example Two (EXT) to a 2.5% Weighting
Trade rationale goes here.

## Model Portfolio: Income
Delete any portfolio with no trades. If no portfolios have trades, delete them all
and the email will say there were no changes to the model portfolios.

## Call to Action: 5i Feature Highlight
Text for this issue's call to action. Links look like [this](https://www.5iresearch.ca).

[button: Button text](https://www.5iresearch.ca)
"""


# --------------------------------------------------------------------------- inline text

def inline_md(text):
    """Minimal markdown: **bold**, *italic*, [text](url). Everything else is escaped."""
    s = html.escape(text, quote=False)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
               lambda m: f'<a href="{m.group(2)}" style="color:{BLUE};font-weight:bold;">{m.group(1)}</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<em>\1</em>", s)
    return s


def p(inner, size=16, color=INK, align="left", margin="0 0 14px", extra=""):
    return (f'<p style="margin:{margin};font-family:{FONT};font-size:{size}px;line-height:1.6;'
            f'color:{color};text-align:{align};{extra}">{inner}</p>')


def md_blocks(text):
    """Paragraphs (blank-line separated), '- ' bullet lists and [button: x](url) lines."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        m = re.fullmatch(r"\[button:\s*(.+?)\]\((https?://\S+)\)", " ".join(lines))
        if m:
            out.append(button(m.group(2), html.escape(m.group(1)), BLUE))
        elif all(ln.startswith(("- ", "* ")) for ln in lines):
            items = "".join(f'<li style="margin:0 0 6px;">{inline_md(ln[2:])}</li>' for ln in lines)
            out.append(f'<ul style="margin:0 0 14px;padding-left:22px;font-family:{FONT};font-size:16px;'
                       f'line-height:1.6;color:{INK};">{items}</ul>')
        else:
            out.append(p(inline_md(" ".join(lines))))
    return "".join(out)


# --------------------------------------------------------------------------- issue.md

def parse_issue(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    head, _, body = raw.partition("\n## ")
    meta = {}
    for line in head.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    sections = []
    for chunk in (body.split("\n## ") if body else []):
        title, _, content = chunk.partition("\n")
        sections.append((title.strip(), content.strip()))
    return meta, sections


def parse_trades(content):
    trades, intro = [], []
    parts = re.split(r"^###\s+", content, flags=re.M)
    intro_text = parts[0].strip()
    for part in parts[1:]:
        title, _, rationale = part.partition("\n")
        rationale = re.sub(r"^\s*Trade Rationale\s*[-–:]\s*", "", rationale.strip(), flags=re.I)
        trades.append((title.strip(), rationale))
    if intro_text:
        intro.append(intro_text)
    return trades, "\n\n".join(intro)


# --------------------------------------------------------------------------- word doc

def _run_html(r):
    text = ""
    for node in r:
        if node.tag == qn("w:t"):
            text += node.text or ""
        elif node.tag == qn("w:tab"):
            text += " "
        elif node.tag in (qn("w:br"), qn("w:cr")):
            text += "\n"
    if not text:
        return "", True
    rpr = r.find(qn("w:rPr"))

    def on(tag):
        el = rpr.find(qn(tag)) if rpr is not None else None
        return el is not None and el.get(qn("w:val")) not in ("0", "false")

    bold = on("w:b")
    s = html.escape(text, quote=False).replace("\n", "<br>")
    if on("w:i"):
        s = f"<em>{s}</em>"
    if bold:
        s = f"<strong>{s}</strong>"
    return s, bold or not text.strip()


def read_docx(path, img_dir, img_prefix):
    d = docx.Document(path)
    blocks, images, title = [], [], ""
    list_items = []

    def flush_list():
        if list_items:
            items = "".join(f'<li style="margin:0 0 6px;">{x}</li>' for x in list_items)
            blocks.append(f'<ul style="margin:0 0 14px;padding-left:22px;font-family:{FONT};font-size:16px;'
                          f'line-height:1.6;color:{INK};">{items}</ul>')
            list_items.clear()

    for para in d.paragraphs:
        el = para._element
        for blip in el.iter(qn("a:blip")):
            rid = blip.get(qn("r:embed"))
            part = para.part.related_parts.get(rid)
            if part is None:
                continue
            ext = os.path.splitext(part.partname)[1].lower() or ".png"
            name = f"figure-{len(images) + 1}{ext}"
            with open(os.path.join(img_dir, name), "wb") as f:
                f.write(part.blob)
            images.append(name)
            flush_list()
            blocks.append(image(f"{img_prefix}/{name}", f"Figure {len(images)}"))

        pieces, all_bold = [], True
        for child in el:
            runs = [child] if child.tag == qn("w:r") else (
                child.findall(qn("w:r")) if child.tag == qn("w:hyperlink") else [])
            chunk = ""
            for r in runs:
                s, b = _run_html(r)
                chunk += s
                all_bold &= b
            if child.tag == qn("w:hyperlink") and chunk:
                target = para.part.rels[child.get(qn("r:id"))].target_ref if child.get(qn("r:id")) else ""
                chunk = f'<a href="{html.escape(target)}" style="color:{BLUE};">{chunk}</a>' if target else chunk
            pieces.append(chunk)
        inner = "".join(pieces).strip()
        text = para.text.strip()
        if not text:
            continue

        if not title and not blocks and text.lower().startswith("market update"):
            title = re.sub(r"^market update\s*[\W_]*\s*", "", text, flags=re.I).strip()
            continue
        is_list = "list" in para.style.name.lower() or el.find(qn("w:pPr") + "/" + qn("w:numPr")) is not None
        if is_list:
            list_items.append(inner)
            continue
        flush_list()
        if re.fullmatch(r"(Figure|Chart|Table)\s+\d+[.:]?.{0,120}", text):
            blocks.append(p(f"<em>{html.escape(text)}</em>", size=13, color=MUTED, align="center",
                            margin="-6px 0 18px"))
        elif all_bold and len(text) <= 140:
            blocks.append(f'<h3 style="margin:22px 0 8px;font-family:{FONT};font-size:19px;line-height:1.35;'
                          f'color:{BLUE};">{html.escape(text)}</h3>')
        else:
            blocks.append(p(inner))
    flush_list()
    return title, "".join(blocks), images


# --------------------------------------------------------------------------- html pieces

# Blog layout: the website's editor draws dotted resize outlines around every
# <table>, so the blog version is built from divs. Email keeps tables for Outlook.
LAYOUT = {"web": False}


def button(url, label, bg):
    if LAYOUT["web"]:
        return (f'<p style="margin:6px 0 18px;text-align:center;"><a href="{html.escape(url)}" target="_blank" '
                f'style="display:inline-block;background:{bg};border-radius:4px;padding:13px 26px;font-family:{FONT};'
                f'font-size:16px;font-weight:bold;color:#ffffff;text-decoration:none;">{label}</a></p>')
    return (f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" align="center" '
            f'style="margin:6px auto 18px;"><tr><td style="background:{bg};border-radius:4px;">'
            f'<a href="{html.escape(url)}" target="_blank" style="display:inline-block;padding:13px 26px;'
            f'font-family:{FONT};font-size:16px;font-weight:bold;color:#ffffff;text-decoration:none;">{label}</a>'
            f'</td></tr></table>')


def image(src, alt):
    return (f'<img src="{html.escape(src)}" alt="{html.escape(alt)}" width="592" '
            f'style="display:block;width:100%;max-width:592px;height:auto;border:0;margin:6px auto 12px;">')


def section(eyebrow, title, body, accent=ORANGE):
    head = (f'<p style="margin:0 0 2px;font-family:{FONT};font-size:12px;font-weight:bold;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:{accent};">{eyebrow}</p>') if eyebrow else ""
    inner = (f'{head}<h2 style="margin:0 0 14px;font-family:{FONT};font-size:26px;line-height:1.25;'
             f'color:{INK};">{title}</h2>{body}')
    if LAYOUT["web"]:
        return f'<div style="padding:26px 0 8px;">{inner}</div>'
    return f'<tr><td style="padding:30px 32px 10px;">{inner}</td></tr>'


def divider():
    if LAYOUT["web"]:
        return f'<hr style="border:0;border-top:1px solid {RULE};margin:6px 0;">'
    return f'<tr><td style="padding:6px 32px;"><div style="border-top:1px solid {RULE};font-size:0;line-height:0;">&nbsp;</div></td></tr>'


def card(body, bg=PANEL, border=RULE, left=None):
    left_css = f"border-left:4px solid {left};" if left else ""
    if LAYOUT["web"]:
        return (f'<div style="background:{bg};border:1px solid {border};{left_css}border-radius:6px;'
                f'padding:18px 20px;margin:0 0 16px;">{body}</div>')
    return (f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="margin:0 0 16px;"><tr><td style="background:{bg};border:1px solid {border};{left_css}'
            f'border-radius:6px;padding:18px 20px;">{body}</td></tr></table>')


def trade_badge(title):
    first = title.split()[0].lower() if title.split() else ""
    for prefixes, label, colour in TRADE_BADGES:
        if first.startswith(prefixes):
            return (f'<span style="display:inline-block;background:{colour};color:#ffffff;font-family:{FONT};'
                    f'font-size:11px;font-weight:bold;letter-spacing:1px;padding:3px 8px;border-radius:3px;'
                    f'margin:0 8px 0 0;vertical-align:2px;">{label}</span>')
    return ""


def portfolio_block(name, content):
    colour = next((c for k, c in PORTFOLIO_COLOURS.items() if k in name.lower()), BLUE)
    trades, intro = parse_trades(content)
    rows = md_blocks(intro) if intro else ""
    for i, (title, rationale) in enumerate(trades):
        sep = f"border-top:1px solid {RULE};" if i else ""
        rows += (f'<div style="{sep}padding:{"14px" if i else "2px"} 0 2px;">'
                 f'<p style="margin:0 0 6px;font-family:{FONT};font-size:17px;font-weight:bold;line-height:1.4;'
                 f'color:{INK};">{trade_badge(title)}{inline_md(title)}</p>'
                 f'{md_blocks(rationale)}</div>')
    head = (f'<p style="margin:0 0 12px;font-family:{FONT};font-size:19px;font-weight:bold;color:{colour};">'
            f'{html.escape(name)} Model Portfolio</p>')
    return card(head + rows, bg="#ffffff", left=colour)


def analytics_plug():
    if LAYOUT["web"]:
        items = "".join(
            f'<p style="margin:0 0 12px;padding-left:26px;text-indent:-26px;font-family:{FONT};font-size:16px;'
            f'line-height:1.5;color:{INK};"><span style="color:{BLUE};font-weight:bold;">&#10003;</span>&nbsp;&nbsp;'
            f'<strong>{t}</strong><br><span style="font-size:15px;color:{MUTED};">{d}</span></p>'
            for t, d in PORTFOLIO_ANALYTICS)
    else:
        items = "".join(
        f'<tr><td valign="top" style="padding:0 10px 12px 0;font-family:{FONT};font-size:18px;color:{BLUE};">&#10003;</td>'
        f'<td style="padding:0 0 12px;"><p style="margin:0;font-family:{FONT};font-size:16px;font-weight:bold;'
        f'color:{INK};">{t}</p><p style="margin:2px 0 0;font-family:{FONT};font-size:15px;line-height:1.5;'
        f'color:{MUTED};">{d}</p></td></tr>'
        for t, d in PORTFOLIO_ANALYTICS)
        items = f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">{items}</table>'
    body = (p("<em>Unlock powerful tools to take your investing to the next level.</em>", color=MUTED)
            + items
            + button(UPGRADE_URL, "Upgrade to Portfolio Analytics", BLUE)
            + p("<strong>Make smarter, data-driven investment decisions and stay ahead of the market.</strong>",
                size=14, align="center", margin="0"))
    return section("Portfolio Analytics", "Deeper market insights",
                   card(body, bg=BLUE_TINT, border="#D3E4F2"), accent=BLUE)


# --------------------------------------------------------------------------- assemble

SECTION_EYEBROWS = {  # key: (eyebrow, title, card accent)
    "report updates": ("Research", "Report Updates", BLUE),
    "new coverage": ("Research", "New Coverage", GREEN),
    "dropping coverage": ("Research", "Dropping Coverage", MUTED),
}


def build_rows(meta, sections, mu_title, mu_html, report_card_src, warnings):
    rows = []
    by_key = {t.lower(): c for t, c in sections}

    research = [k for k in ("report updates", "new coverage", "dropping coverage") if by_key.get(k)]
    for i, key in enumerate(research):
        eyebrow, title, accent = SECTION_EYEBROWS[key]
        body = card(md_blocks(by_key[key]), left=accent)
        if i == len(research) - 1:
            body += button(REPORTS_URL, "Read the latest reports", BLUE)
        rows.append(section(eyebrow, title, body))
        rows.append(divider())
    if not research:
        warnings.append("No Report Updates / Coverage sections in issue.md.")

    known = set(SECTION_EYEBROWS) | {"market intro"}
    for t, c in sections:
        tl = t.lower()
        if tl in known or tl.startswith(("model portfolio", "call to action")):
            continue
        rows.append(section("", html.escape(t), md_blocks(c)))
        rows.append(divider())

    intro = md_blocks(by_key["market intro"]) if by_key.get("market intro") else ""
    if not intro:
        warnings.append("No Market Intro in issue.md.")
    if mu_html or intro:
        rows.append(section("Market Update", html.escape(mu_title or "Market Update"), intro + mu_html))
        rows.append(divider())

    if report_card_src:
        rows.append(section("Macro", "Economic Report Card", image(report_card_src, "Macro Economic Report Card")))
        rows.append(divider())
    else:
        warnings.append("No Report Card.png in the issue folder.")

    portfolios = [(t.split(":", 1)[1].strip() if ":" in t else "Model", c)
                  for t, c in sections if t.lower().startswith("model portfolio")]
    portfolios = [(n, c) for n, c in portfolios if re.search(r"^###", c, flags=re.M)]
    if portfolios:
        body = "".join(portfolio_block(n, c) for n, c in portfolios)
    else:
        body = card(p("No changes to the model portfolios this period.", margin="0", align="center"))
    rows.append(section("Model Portfolios", "Model Portfolio Changes", body, accent=BLUE))
    rows.append(divider())

    rows.append(analytics_plug())

    cta = [(t, c) for t, c in sections if t.lower().startswith("call to action")]
    if cta:
        t, c = cta[0]
        title = t.split(":", 1)[1].strip() if ":" in t else "From 5i"
        rows.append(section("From 5i", html.escape(title),
                            card(md_blocks(c), bg=ORANGE_TINT, border="#F0DCC6", left=ORANGE)))
    else:
        warnings.append("No Call to Action section in issue.md.")

    rows.append(f'<tr><td style="padding:14px 32px 26px;"><p style="margin:0;font-family:{FONT};font-size:17px;'
                f'color:{INK};">With precision and insight,</p><p style="margin:2px 0 0;font-family:{FONT};'
                f'font-size:24px;font-weight:bold;color:{DBLUE};">The 5i Research Team</p></td></tr>')
    return "".join(rows)


def build_email(meta, rows, date_label):
    social = "".join(
        f'<a href="{u}" target="_blank" style="display:inline-block;margin:0 6px;">'
        f'<img src="{SOCIAL_ICON.format(icon)}" width="32" height="32" alt="{name}" style="border:0;"></a>'
        for name, u, icon in SOCIAL)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="X-UA-Compatible" content="IE=edge"><title>*|MC:SUBJECT|*</title>
<style>
body{{margin:0;padding:0;}} img{{-ms-interpolation-mode:bicubic;}}
a[x-apple-data-detectors]{{color:inherit!important;text-decoration:none!important;}}
@media only screen and (max-width:480px){{ .px td{{padding-left:18px!important;padding-right:18px!important;}} h2{{font-size:23px!important;}} }}
</style></head>
<body style="margin:0;padding:0;background:#EEF1F4;">
<span style="display:none;font-size:0;line-height:0;max-height:0;max-width:0;opacity:0;overflow:hidden;mso-hide:all;">*|MC_PREVIEW_TEXT|*</span>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#EEF1F4;">
<tr><td align="center" style="padding:20px 10px;">
<p style="margin:0 0 10px;font-family:{FONT};font-size:12px;color:{MUTED};"><a href="*|ARCHIVE|*" style="color:{MUTED};">View this email in your browser</a></p>
<table role="presentation" width="660" cellpadding="0" cellspacing="0" border="0" class="px" style="width:100%;max-width:660px;background:#ffffff;border-radius:8px;overflow:hidden;">
<tr><td style="height:6px;background:{BLUE};font-size:0;line-height:0;">&nbsp;</td></tr>
<tr><td align="center" style="padding:28px 32px 18px;border-bottom:3px solid {ORANGE};">
<img src="{LOGO_URL}" width="240" alt="5i Research" style="display:block;width:240px;max-width:70%;height:auto;border:0;margin:0 auto 12px;">
<p style="margin:0;font-family:{FONT};font-size:13px;font-weight:bold;letter-spacing:2px;text-transform:uppercase;color:{BLUE};">Market, Portfolio &amp; Report Updates</p>
<p style="margin:4px 0 0;font-family:{FONT};font-size:14px;color:{MUTED};">{date_label}</p>
</td></tr>
<tr><td mc:edit="body" style="padding:0;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
{rows}
</table></td></tr>
<tr><td align="center" style="padding:18px 32px;background:{PANEL};border-top:1px solid {RULE};">{social}</td></tr>
<tr><td style="padding:18px 32px 26px;background:{PANEL};font-family:{FONT};font-size:11px;line-height:1.55;color:{MUTED};">
<p style="margin:0 0 10px;"><em>Copyright &copy; *|CURRENT_YEAR|* 5i Research Inc. All rights reserved.</em><br>
You're receiving this email because you opted in to receive information from our website 5iResearch.ca.</p>
<p style="margin:0 0 10px;">{DISCLAIMER}</p>
<p style="margin:0 0 10px;">*|LIST:ADDRESSLINE|*</p>
<p style="margin:0;"><a href="*|UPDATE_PROFILE|*" style="color:{MUTED};">Update your preferences</a> &middot; <a href="*|UNSUB|*" style="color:{MUTED};">Unsubscribe</a></p>
</td></tr>
</table>
</td></tr></table>
</body></html>
"""


def build_post(rows):
    return f'<div class="market-update" style="max-width:660px;margin:0 auto;">\n{rows}\n</div>\n'


def issue_date(folder, meta):
    if meta.get("date"):
        return datetime.strptime(meta["date"], "%Y-%m-%d")
    name, parent_year = os.path.basename(folder), None
    for part in reversed(os.path.normpath(folder).split(os.sep)):
        if re.fullmatch(r"20\d\d", part):
            parent_year = int(part)
            break
    for fmt in ("%b %d", "%B %d"):
        try:
            d = datetime.strptime(name.replace("Sept", "Sep"), fmt)
            return d.replace(year=parent_year or datetime.now().year)
        except ValueError:
            pass
    return datetime.now()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    folder = os.path.abspath(sys.argv[1].strip('"'))
    issue_md = os.path.join(folder, "issue.md")
    if not os.path.exists(issue_md):
        with open(issue_md, "w", encoding="utf-8") as f:
            f.write(ISSUE_TEMPLATE)
        print(f"Created {issue_md}\nFill it in, then run this again.")
        os.startfile(issue_md) if hasattr(os, "startfile") else None
        return

    meta, sections = parse_issue(issue_md)
    when = issue_date(folder, meta)
    date_str = when.strftime("%Y-%m-%d")
    date_label = f"{when:%B} {when.day}, {when.year}"
    build_dir = os.path.join(folder, "_build")
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)
    img_prefix = f"{PAGES_BASE}/{date_str}"
    warnings = []

    docs = [d for d in glob.glob(os.path.join(folder, "Market Update*.docx"))
            if not os.path.basename(d).startswith("~$")]
    docs.sort(key=lambda d: ("final" not in d.lower(), -os.path.getmtime(d)))
    mu_title, mu_html, images = "", "", []
    if docs:
        mu_title, mu_html, images = read_docx(docs[0], build_dir, img_prefix)
        if len(docs) > 1:
            warnings.append(f"Several Market Update docs found; used {os.path.basename(docs[0])}.")
    else:
        warnings.append("No 'Market Update - ....docx' in the issue folder.")

    report_card_src = ""
    rc = os.path.join(folder, "Report Card.png")
    if os.path.exists(rc):
        shutil.copyfile(rc, os.path.join(build_dir, "report-card.png"))
        images.append("report-card.png")
        report_card_src = f"{img_prefix}/report-card.png"

    rows = build_rows(meta, sections, mu_title, mu_html, report_card_src, warnings)
    email_html = build_email(meta, rows, date_label)
    LAYOUT["web"] = True
    try:
        post_html = build_post(build_rows(meta, sections, mu_title, mu_html, report_card_src, []))
    finally:
        LAYOUT["web"] = False
    subject = meta.get("subject") or "Market, Model Portfolio, and Report Updates!"
    preview = meta.get("preview") or mu_title
    blog_title = meta.get("blog title") or (f"Market Update: {mu_title}" if mu_title else subject)

    for name, content in (("email.html", email_html), ("post.html", post_html)):
        with open(os.path.join(build_dir, name), "w", encoding="utf-8") as f:
            f.write(content)

    host_dir = os.path.join(IMAGES_ROOT, date_str)
    os.makedirs(host_dir, exist_ok=True)
    for name in images:
        shutil.copyfile(os.path.join(build_dir, name), os.path.join(host_dir, name))

    data = {"subject": subject, "preview": preview, "blog_title": blog_title, "post": post_html,
            "email": email_html, "pages_prefix": img_prefix + "/", "warnings": warnings,
            "check_image": f"{img_prefix}/{images[0]}" if images else ""}
    with open(KIT_TEMPLATE, encoding="utf-8") as f:
        page = f.read()
    for key, val in {"DATE": html.escape(date_label), "TITLE": html.escape(mu_title or subject),
                     "DATA_JSON": json.dumps(data).replace("</", "<\\/")}.items():
        page = page.replace(f"%%{key}%%", val)
    kit = os.path.join(build_dir, "kit.html")
    with open(kit, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"Built {date_label}: {len(images)} image(s), {len(sections)} issue.md section(s)")
    for w in warnings:
        print(f"  ! {w}")
    print(f"Images copied to {host_dir} - commit & push them so they load in Mailchimp.")
    print(f"Kit: {kit}")
    if "--no-open" not in sys.argv:
        webbrowser.open("file:///" + kit.replace(os.sep, "/"))


if __name__ == "__main__":
    main()
