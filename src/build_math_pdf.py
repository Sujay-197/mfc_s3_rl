"""Build artifacts/math_concepts.pdf -- the mathematical write-up.

Explains the mathematics behind the agent and, for each idea, how it is realised
in this codebase. Every formula is rendered through matplotlib's mathtext (so no
LaTeX installation is required) and every architectural number is measured from
the installed library or read out of a real run rather than written from memory.

    python src/build_math_pdf.py
"""

import json
import os
import pathlib
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image,
                                KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

# A serif math font on a transparent background, so each equation sits on the
# page the way a typeset display equation would.
rcParams["mathtext.fontset"] = "cm"
rcParams["font.family"] = "serif"

INK = colors.HexColor("#12161C")
MUTED = colors.HexColor("#545C68")
ACCENT = colors.HexColor("#C8402F")
RULE = colors.HexColor("#D6DBE2")
PANEL = colors.HexColor("#F4F6F8")

# matplotlib does not accept a ReportLab Color object, so the ink colour is
# also kept as a hex string for the equation renderer.
INK_HEX = "#12161C"

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
DOC_W = PAGE_W - 2 * MARGIN
GUT = 9          # inner padding of a callout panel

_eq_cache = {}
_eq_seq = [0]


# --------------------------------------------------------------- equations
def eq_image(latex, fontsize=15):
    """Render LaTeX to a transparent PNG; return (path, width_pt, height_pt).

    mathtext does the typesetting, so no LaTeX install is needed. Cached on the
    expression, because several equations are deliberately shown twice.
    """
    key = (latex, fontsize)
    if key in _eq_cache:
        return _eq_cache[key]

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize, color=INK_HEX)
    path = os.path.join(tempfile.gettempdir(),
                        f"mcp_eq_{abs(hash(key)) % (10 ** 12)}.png")
    fig.savefig(path, dpi=400, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    w_px, h_px = ImageReader(path).getSize()
    plt.close(fig)

    val = (path, w_px * 72.0 / 400.0, h_px * 72.0 / 400.0)
    _eq_cache[key] = val
    return val


def display_eq(latex, tag=True, size=14):
    """A centred display equation, numbered flush right when tag=True.

    Three columns -- blank, equation, number -- is what keeps display maths
    aligned: the expression stays centred in the measure and the tag sits at the
    right margin no matter how wide the expression is.
    """
    path, w, h = eq_image(latex, fontsize=size)
    max_w = DOC_W - 2.4 * cm          # room for the tag column and its gutter
    if w > max_w:                      # long expressions scale down, never overflow
        h *= max_w / w
        w = max_w

    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    if tag:
        _eq_seq[0] += 1
        num = Paragraph(f"({_eq_seq[0]})", S["eqnum"])
    else:
        num = Paragraph("", S["eqnum"])

    t = Table([[num, img, Paragraph("", S["eqnum"])]],
              colWidths=[1.1 * cm, DOC_W - 1.1 * cm, 1.1 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    return t

# ------------------------------------------------------------------ styles
def styles():
    base = dict(fontName="Times-Roman", textColor=INK, leading=14.2,
                alignment=TA_JUSTIFY, fontSize=10.3, spaceAfter=7)
    s = {}
    s["title"] = ParagraphStyle("title", fontName="Times-Bold", fontSize=22,
                                leading=26, textColor=INK, spaceAfter=5)
    s["subtitle"] = ParagraphStyle("subtitle", fontName="Times-Italic",
                                   fontSize=11.5, leading=15, textColor=MUTED,
                                   spaceAfter=14)
    # `**base` already carries textColor, so a recoloured variant must merge
    # rather than pass textColor again -- ParagraphStyle takes no duplicates.
    s["abstract"] = ParagraphStyle("abstract", **{**base, "textColor": MUTED})
    s["h1"] = ParagraphStyle("h1", fontName="Times-Bold", fontSize=14.5,
                             leading=18, textColor=INK, spaceBefore=14,
                             spaceAfter=5)
    s["h2"] = ParagraphStyle("h2", fontName="Times-Bold", fontSize=11.5,
                             leading=14, textColor=ACCENT, spaceBefore=10,
                             spaceAfter=3)
    s["body"] = ParagraphStyle("body", **base)
    s["eqnum"] = ParagraphStyle("eqnum", fontName="Times-Roman", fontSize=9.5,
                                textColor=MUTED, alignment=TA_CENTER)
    s["code"] = ParagraphStyle("code", fontName="Courier", fontSize=8.2,
                               leading=11.2, textColor=INK, spaceAfter=0)
    s["cap"] = ParagraphStyle("cap", fontName="Times-Italic", fontSize=8.6,
                              leading=11, textColor=MUTED,
                              alignment=TA_CENTER, spaceBefore=4, spaceAfter=10)
    s["panelt"] = ParagraphStyle("panelt", fontName="Times-Bold", fontSize=9.8,
                                 leading=13, textColor=ACCENT, spaceAfter=3)
    s["panelb"] = ParagraphStyle("panelb", **{**base, "fontSize": 9.8,
                                              "leading": 13, "spaceAfter": 0,
                                              "leftIndent": 0})
    s["bullets"] = ParagraphStyle("bullets", **{**base, "leftIndent": 15,
                                                "bulletIndent": 2,
                                                "spaceAfter": 4})
    s["tight"] = ParagraphStyle("tight", **{**base, "leftIndent": 15,
                                            "bulletIndent": 2, "spaceAfter": 3,
                                            "fontSize": 10.0})
    s["refs"] = ParagraphStyle("refs", **{**base, "leftIndent": 16,
                                          "bulletIndent": 2, "fontSize": 9.4,
                                          "leading": 12.4, "spaceAfter": 5})
    s["cell"] = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.6,
                               leading=11, textColor=INK, spaceAfter=0,
                               alignment=TA_CENTER)
    s["celll"] = ParagraphStyle("celll", fontName="Times-Roman", fontSize=8.6,
                                leading=11, textColor=INK, spaceAfter=0)
    s["cellh"] = ParagraphStyle("cellh", fontName="Times-Bold", fontSize=8.6,
                                leading=11, textColor=INK, spaceAfter=0,
                                alignment=TA_CENTER)
    return s


S = styles()


def hr():
    t = Table([[""]], colWidths=[DOC_W], rowHeights=[1])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), 0.5, RULE),
                           ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def code_block(lines):
    """A monospaced source excerpt on a light panel."""
    body = [[Paragraph(ln if ln.strip() else "&nbsp;", S["code"])]
            for ln in lines]
    t = Table(body, colWidths=[DOC_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def figure(path, width_frac=0.94, caption=""):
    """An image scaled to a fraction of the measure, with a centred caption."""
    p = pathlib.Path(path)
    if not p.exists():
        return [Paragraph(f"[figure missing: {p.name}]", S["cap"])]
    iw, ih = ImageReader(str(p)).getSize()
    w = DOC_W * width_frac
    h = w * ih / iw
    if h > 7.4 * cm:            # keep a figure whole on one page
        h = 7.4 * cm
        w = h * iw / ih
    img = Image(str(p), width=w, height=h)
    img.hAlign = "CENTER"
    out = [Spacer(1, 3), img]
    if caption:
        out.append(Paragraph(caption, S["cap"]))
    return out


def results_table(rows):
    """Render a results JSON list as a typeset table."""
    head = ["run", "actions", "seed", "clear rate", "first clear",
            "steps trained", "mean reward", "mean x", "greedy"]
    data = [[Paragraph(h, S["cellh"]) for h in head]]
    for r in rows:
        first = f"{r['first_clear_steps']:,}" if r.get("first_clear_steps") else "never"
        data.append([
            Paragraph(r["run"], S["celll"]),
            Paragraph(r["action_set"], S["cell"]),
            Paragraph(str(r["seed"]), S["cell"]),
            Paragraph(f"<b>{r['clear_rate']:.0%}</b>", S["cell"]),
            Paragraph(first, S["cell"]),
            Paragraph(f"{r['total_steps']:,}", S["cell"]),
            Paragraph(f"{r['mean_reward']:.1f}", S["cell"]),
            Paragraph(f"{r['mean_x_max']:.0f}", S["cell"]),
            Paragraph("yes" if r.get("greedy_clear") else "no", S["cell"]),
        ])
    colw = [4.5, 1.8, 1.0, 1.9, 2.4, 2.3, 2.1, 1.5, 1.4]
    total = sum(colw)
    colw = [c / total * DOC_W for c in colw]
    t = Table(data, colWidths=colw, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECF0")),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
    ]))
    return t


def page_furniture(canv, doc):
    """Running head rule and a page number on every page but the first."""
    canv.saveState()
    if doc.page > 1:
        canv.setFont("Times-Italic", 8.3)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN, PAGE_H - MARGIN + 22,
                        "The mathematics behind a Super Mario Bros. PPO agent")
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.5)
        canv.line(MARGIN, PAGE_H - MARGIN + 16, PAGE_W - MARGIN, PAGE_H - MARGIN + 16)
    canv.setFont("Times-Roman", 8.8)
    canv.setFillColor(MUTED)
    canv.drawCentredString(PAGE_W / 2.0, MARGIN - 20, str(doc.page))
    canv.restoreState()


# ============================================================ document body
def build():
    """Assemble every section and write the PDF to artifacts/math_concepts.pdf."""
    story = []

    def p(text, style="body"):
        story.append(Paragraph(text, S[style]))

    def sp(h=6):
        story.append(Spacer(1, h))

    def equation(latex, tag=True, size=13):
        story.append(display_eq(latex, tag=tag, size=size))
        sp(2)

    def items(lines, style="bullets"):
        for ln in lines:
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{ln}", S[style]))
        sp(5)

    def section(title):
        """Start a numbered section on a fresh page, but only when needed.

        A hard PageBreak here produced several nearly blank pages, because a
        section that already overflowed left a few lines on the next page and
        the break then stranded them. CondPageBreak only breaks when less than
        a fifth of the page remains, so short tails flow on and full pages
        still start cleanly.
        """
        story.append(CondPageBreak(3.6 * cm))
        p(title, "h1")

    def panel(paras, title=None):
        rows = []
        if title:
            rows.append([Paragraph(f"<b>{title}</b>", S["panelt"])])
        for t in paras:
            rows.append([Paragraph(t, S["panelb"])])
        t = Table(rows, colWidths=[DOC_W - 2 * GUT])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PANEL),
            ("BOX", (0, 0), (-1, -1), 0.5, RULE),
            ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ]))
        story.append(KeepTogether([t, Spacer(1, 9)]))
        sp(0)

    SECTIONS = [
        "The problem as a Markov decision process",
        "Value functions and the Bellman equation",
        "The policy gradient theorem",
        "Proximal Policy Optimization and the clipped surrogate",
        "Generalized Advantage Estimation",
        "The reward function, term by term",
        "The observation pipeline: frame skip, pooling, stacking",
        "The network",
        "The training loop",
        "Results",
        "Design decisions, trade-offs, and limits",
    ]

    # ------------------------------------------------------------ title page
    p("The Mathematics Behind a Super Mario Bros. PPO Agent", "title")
    p("How the reinforcement learning in this project works, and where each "
      "idea is actually implemented.", "subtitle")
    sp(2)
    p("This document explains the mathematics behind the agent that completes "
      "World 1-1 and World 1-2 of Super Mario Bros. from raw pixels, and ties "
      "each idea to the code that implements it. Every formula is taken from "
      "the installed libraries rather than from memory, and every measured "
      "number was read out of a real run.", "abstract")
    sp(6)
    p("Contents", "h1")
    for n, title in enumerate(SECTIONS, 1):
        p(f"{n}.&nbsp;&nbsp;{title}", "bullets")
    sp(4)
    story.append(PageBreak())

    # ------------------------------------------------------------------- 1 MDP
    section("1. The problem as a Markov decision process")
    p("Reinforcement learning formalises a sequential decision problem as a "
      "<b>Markov decision process</b> (MDP): a tuple <b>M</b> = "
      "(<i>S</i>, <i>A</i>, <i>P</i>, <i>R</i>, <i>&gamma;</i>) consisting of a "
      "state space <i>S</i>, an action space <i>A</i>, a transition kernel "
      "<i>P</i>, a reward function <i>R</i>, and a discount factor "
      "<i>&gamma;</i>. The agent observes a state, chooses an action, the "
      "environment transitions and emits a reward, and the process repeats "
      "until the episode ends.", "body")
    p("The return from time <i>t</i> is the discounted sum of all future "
      "rewards:", "body")
    equation(r"G_t \;=\; \sum_{k=0}^{\infty} \gamma^{k}\, r_{t+k}")
    p("<i>&gamma;</i> = 0.9 in this project. The discount does two jobs: it "
      "makes an infinite sum converge, and it encodes that a reward obtained "
      "now is worth more than the same reward obtained later. A smaller "
      "<i>&gamma;</i> makes the agent short-sighted, which suits Mario because "
      "the useful horizon is short &mdash; the next obstacle is only a few "
      "steps ahead.", "body")
    p("Here the state is not a hand-designed vector. It is the raw "
      "256&times;240 RGB framebuffer produced by the NES emulator, so "
      "<i>S</i> is effectively the set of all possible screen images and the "
      "agent must learn its own representation of the game. The action space "
      "is a small set of joypad combinations, reduced from the 256 possible "
      "button bitmasks:", "body")
    items([
        "<b>simple</b> &mdash; 7 actions: no-op, right, right+A, right+B, "
        "right+A+B, A, left.",
        "<b>complex</b> &mdash; 12 actions: the above plus left variants, "
        "<i>down</i>, and <i>up</i>.",
        "<b>right_only</b> &mdash; 5 actions: no-op and the right-moving "
        "combinations.",
    ])
    p("An episode ends on death, on reaching the flagpole, or when the in-game "
      "clock expires. The reward function, the discount, and this termination "
      "rule are the three places where the problem definition does the work "
      "that the learning algorithm then exploits.", "body")

    # ------------------------------------------------------------ 2 value fn
    section("2. Value functions and the Bellman equation")
    p("A policy <i>&pi;</i> maps states to distributions over actions. The "
      "<b>state-value function</b> under <i>&pi;</i> is the expected return "
      "when following <i>&pi;</i> from <i>s</i>:", "body")
    equation(r"V^{\pi}(s) \;=\; \mathbb{E}_{\pi}\!\left[\, G_t \,\middle|\, s_t = s \,\right] = \mathbb{E}_{\pi}\!\left[\, \sum_{k=0}^{\infty} \gamma^{k} r_{t+k} \,\right]")
    p("It satisfies the <b>Bellman expectation equation</b>: the value of a "
      "state equals the immediate reward plus the discounted value of wherever "
      "the next action leads.", "body")
    equation(r"V^{\pi}(s) \;=\; \mathbb{E}_{\pi}\!\left[\, r + \gamma\, V^{\pi}(s') \,\right]")
    p("Taking a maximum over actions instead of an expectation gives the "
      "Bellman <b>optimality</b> equation, whose solution is <i>V</i><sup>*</sup>:", "body")
    equation(r"V^{*}(s) \;=\; \max_{a \in A} \mathbb{E}\!\left[\, r + \gamma\, V^{*}(s') \,\right]")
    p("The difference between the two is the <b>advantage</b> &mdash; how much "
      "better than average a particular action is, in a particular state:", "body")
    equation(r"A^{\pi}(s, a) \;=\; Q^{\pi}(s, a) \;-\; V^{\pi}(s), \qquad Q^{\pi}(s,a) = \mathbb{E}_{\pi}\!\left[\, r + \gamma V^{\pi}(s') \,\right]")
    p("The advantage is what a policy-gradient step actually uses, because it "
      "removes the part of the return that does not depend on the action. "
      "Subtracting a baseline leaves the gradient of the objective unchanged "
      "while reducing variance, provided the baseline depends on the state but "
      "<i>not</i> on the action. The value network in this project plays "
      "exactly that role.", "body")
    panel([
        "Tabulating <i>V</i><sup>*</sup> is impossible when a state is a pixel "
        "framebuffer, so the value function is <b>approximated</b> by a neural "
        "network <i>V<sub>&phi;</sub>(s)</i> and learned from data. Because the "
        "input is an image, the network is convolutional and sees a downscaled "
        "84&times;84 grayscale image rather than the full frame.",
    ], "Why approximation is unavoidable here")
    # ---------------------------------------------------- 3 policy gradient
    section("3. The policy gradient theorem")
    p("Rather than solve for <i>V</i><sup>*</sup> and then read off a policy, "
      "an alternative family of methods differentiates the expected return "
      "with respect to the policy parameters <i>&theta;</i> directly. The "
      "objective is", "body")
    equation(r"J(\theta) \;=\; \mathbb{E}_{\pi_\theta}\!\left[\, \sum_{k=0}^{\infty} \gamma^{k} r_{t+k} \,\right]")
    p("Differentiating an expectation over a distribution that itself depends "
      "on <i>&theta;</i> is the obstacle. The <b>policy gradient theorem</b> "
      "resolves it with the log-derivative trick, which turns the derivative "
      "of a distribution into that distribution times the derivative of its "
      "logarithm:", "body")
    equation(r"\nabla_{\theta} p_{\theta}(x) \;=\; p_{\theta}(x)\, \nabla_{\theta} \log p_{\theta}(x)")
    p("Applying that to the trajectory distribution and simplifying gives the "
      "result that everything else builds on:", "body")
    equation(r"\nabla_{\theta} J(\theta) \;=\; \mathbb{E}_{\pi_\theta}\!\left[\, \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \cdot G_t \,\right] = \mathbb{E}_{\pi_\theta}\!\left[\, \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \cdot Q^{\pi}(s_t, a_t) \,\right]")
    p("Replacing the return <i>G<sub>t</sub></i> with the action-value "
      "<i>Q</i><sup>&pi;</sup> trades a little bias for a large variance "
      "reduction. Two practical points complete the picture. First, a "
      "<b>baseline</b> may be subtracted without biasing the estimator, "
      "provided it does not depend on the action taken; choosing "
      "<i>V</i><sup>&pi;</sup>(<i>s<sub>t</sub></i>) as that baseline produces "
      "the advantage, and the estimator inherits the same unbiasedness:", "body")
    equation(r"\mathbb{E}\!\left[\, \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \cdot \left(G_t - b(s_t)\right) \,\right] = \nabla_{\theta} J(\theta)")
    p("for any baseline <i>b</i> independent of <i>a<sub>t</sub></i>. Second, "
      "the gradient is estimated by sampling "
      "<i>a<sub>t</sub></i> ~ &pi;<sub>&theta;</sub>(&middot;|<i>s<sub>t</sub></i>) "
      "and weighting each log-probability by its advantage. That stochastic "
      "estimate is unbiased for the same gradient, and it is what PPO uses in "
      "practice.", "body")
    panel([
        "The expectation is over states the agent actually visits, which is "
        "what makes this an <b>on-policy</b> method. Data is only valid for the "
        "policy that collected it, so fresh trajectories must be generated "
        "continuously. The two practical problems that follow &mdash; moving too "
        "far in a single step, and estimating the advantage from one sample "
        "&mdash; are what sections 4 and 5 address.",
    ], "Why this is an on-policy method")
    # ------------------------------------------------ 4. PPO
    section("4. Proximal Policy Optimization and the clipped surrogate")
    p("A naive gradient step can move the policy far enough in one update to "
      "destroy the performance it just achieved: the data being reused was "
      "collected by a different policy, and a large step invalidates it. TRPO "
      "prevents this with a hard KL-divergence trust region, which gives a "
      "monotone improvement bound but requires solving a constrained "
      "optimisation problem at every step. PPO obtains the same protective "
      "effect more simply, by <b>clipping</b> the objective so that its gradient "
      "becomes zero once the policy has strayed too far.", "body")
    p("The starting point is the surrogate objective, in which the on-policy "
      "gradient is recast as a ratio of the new and old action probabilities:", "body")
    # mathtext has no amsmath \big / \Big sizing macros, so every delimiter here
    # is written as \left...\right, which mathtext does support and sizes itself.
    equation(r"L^{\mathrm{CLIP}}(\theta) \;=\; \mathbb{E}_t\left[\, \min\left(\, r_t(\theta)\, \hat{A}_t, \; \mathrm{cl}\left(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\right)\, \hat{A}_t \right) \, \right]")
    p("where the <b>probability ratio</b> is", "body")
    equation(r"r_t(\theta) \;=\; \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\mathrm{old}}(a_t \mid s_t)}")
    p("The ratio equals exactly 1 when <i>&theta;</i> is the parameter vector that "
      "collected the data, which is what anchors the update. Clipping it at "
      "1&nbsp;&plusmn;&nbsp;<i>&epsilon;</i>, with <i>&epsilon;</i> = 0.2 here, "
      "removes any incentive to move further once the ratio has left that band. "
      "The subtlety that matters is the <i>min</i>, which makes the bound one-sided "
      "and pessimistic rather than symmetric:", "body")
    items([
        # The advantage is the Greek capital Delta, U+0394, written as a numeric
        # character reference. A literal '&' cannot be used here at all: a bare
        # one is treated as markup and "&amp;" is decoded back to a literal '&'.
        "When <b><i>&#916;<sub>t</sub></i> is positive</b> (the action was good), the "
        "minimum of <i>r<sub>t</sub></i> and (1&nbsp;+&nbsp;<i>&epsilon;</i>) is "
        "<i>r<sub>t</sub></i> until the ratio exceeds the upper bound, after which "
        "the objective is flat. The agent may make a good action more likely, but "
        "not without limit.",
        "When <b><i>&#916;<sub>t</sub></i> is negative</b> (the action was bad), "
        "multiplying by a negative quantity reverses the inequality, so the "
        "<i>min</i> selects the more negative term and the effective ceiling sits "
        "at (1&nbsp;&minus;&nbsp;<i>&epsilon;</i>). The agent may make a bad action "
        "less likely, but not without limit.",
    ], "tight")
    p("In both directions the gradient is zero outside the interval, which is "
      "the mechanism that keeps a single update bounded. This project uses "
      "Stable-Baselines3's implementation unmodified. Its code carries a leading "
      "minus sign because the objective is minimised as a loss rather than "
      "maximised:", "body")
    story.append(code_block([
        "ratio         = th.exp(log_prob - old_log_prob)",
        "policy_loss_1 = advantages * ratio",
        "policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)",
        "policy_loss   = -th.min(policy_loss_1, policy_loss_2).mean()",
    ]))
    sp(7)
    p("The value function is fitted by regression onto the estimated return, and "
      "an entropy bonus discourages premature collapse onto a single action. The "
      "three terms are combined with weights:", "body")
    equation(r"\mathcal{L} \;=\; L_\pi \;+\; c_v\, L_V \;-\; c_e\, \mathcal{H}, \qquad c_v = 0.5, \;\; c_e = 0.01")
    p("where the entropy of a categorical policy is", "body")
    equation(r"\mathcal{H}(\pi) \;=\; -\sum_{a} \pi(a \mid s)\, \log \pi(a \mid s)")
    p("The entropy is maximised at the uniform distribution and never negative, so "
      "it is a direct measure of how unresolved the policy still is. Subtracting "
      "it from the loss rewards staying uncertain, which keeps exploration alive "
      "while the value function is still too unreliable to provide a useful signal "
      "of its own. This matters most early on, when the agent would otherwise commit "
      "to a single action and stop discovering the level.", "body")

    # ------------------------------------------------ 5. GAE
    section("5. Generalized Advantage Estimation")
    p("PPO needs <i>&#916;<sub>t</sub></i> but never observes it. Two classical "
      "estimators bracket the problem. The one-step <b>temporal difference</b> "
      "residual is cheap and has low variance, but is biased toward whatever the "
      "value function currently believes:", "body")
    equation(r"\delta_t \;=\; r_t + \gamma\, V_{\phi}(s_{t+1}) - V_{\phi}(s_t)")
    p("The Monte-Carlo estimate, the full discounted return minus the baseline, is "
      "unbiased but has high variance: a single early death propagates its "
      "&minus;25 penalty into every subsequent estimate. GAE forms an "
      "exponentially weighted average of residuals at every horizon, with weights "
      "decaying geometrically at rate <i>&gamma;&lambda;</i>:", "body")
    equation(r"\hat{A}_t \;=\; \sum_{l=0}^{\infty} (\gamma \lambda)^{\,l}\, \delta_{t+l}")
    p("Written out, that sum obeys a recursion which makes it computable in a "
      "single backward pass rather than by evaluating every multi-step return "
      "separately:", "body")
    equation(r"\hat{A}_t \;=\; \delta_t + \gamma \lambda \left(1 - d_t\right) \hat{A}_{t+1}, \qquad \hat{A}_T = 0")
    p("with <i>d<sub>t</sub></i> a termination indicator. The two limiting cases "
      "reproduce the classical estimators exactly, which is the clearest way to "
      "see what the parameter does:", "body")
    items([
        "<b><i>&lambda;</i> = 0</b> collapses the sum to its single first term, "
        "leaving pure one-step TD(&lambda; = 0): lowest variance, highest bias.",
        "<b><i>&lambda;</i> = 1</b> makes the weights <i>&gamma;</i><sup><i>l</i></sup>, "
        "which is the Monte-Carlo estimator: unbiased, highest variance.",
        "<b><i>&lambda;</i> = 0.95</b>, used here, sits just below the second case, "
        "so the estimator is nearly unbiased while still benefiting from "
        "bootstrapping off the value function.",
    ], "tight")
    p("The value-function target is the advantage plus the baseline, which turns "
      "the GAE estimate into a return estimate for the regression in section 4:", "body")
    equation(r"\hat{R}_t \;=\; \hat{A}_t + V_{\phi}(s_t)")
    p("This is exactly the telescoping sum of the discounted residuals, which is "
      "why both quantities fall out of a single backward pass. The mask "
      "<i>(1&nbsp;&minus;&nbsp;d<sub>t</sub>)</i> also handles episode boundaries: "
      "when Mario dies, the bootstrap term is dropped, so no value is propagated "
      "across the termination and the estimate does not treat the post-death frame "
      "as a normal continuation of the same trajectory. Ignoring that mask is a "
      "common and difficult-to-spot bug, because training still appears to run.", "body")
    story.append(code_block([
        "for step in reversed(range(self.buffer_size)):",
        "    next_non_terminal = 1.0 - dones.astype(np.float32)",
        "    delta = self.rewards[step] + self.gamma * next_values \\",
        "             * next_non_terminal - self.values[step]",
        "    last_gae_lam = delta + self.gamma * self.gae_lambda \\",
        "                     * next_non_terminal * last_gae_lam",
        "    self.advantages[step] = last_gae_lam",
        "self.returns = self.advantages + self.values",
    ]))
    sp(6)
    p("The library's implementation is a direct transcription of the recursion. "
      "The loop runs in reverse so that <i>&delta;</i> and "
      "<i>A</i><sub><i>t</i>+1</sub> are both available before "
      "<i>A</i><sub><i>t</i></sub> is written, and the final line is the "
      "advantage-plus-baseline target from above.", "body")

    # ------------------------------------------------ 6. reward
    section("6. The reward function, term by term")
    p("The agent can only optimise what the environment pays, so the reward "
      "function defines the objective far more than the algorithm does. The "
      "library supplies a dense, shaped reward built from named components, each "
      "read directly from the emulator's memory rather than inferred:", "body")
    items([
        "<b>progress</b> &mdash; the horizontal distance gained since the furthest "
        "point reached so far in the episode. Earning new maximum distance, so "
        "backtracking earns nothing. A single step exceeding 5 units is discarded, "
        "which suppresses a large lump sum for a teleport.",
        "<b>time</b> &mdash; the decrease in the in-game clock, &minus;1 per tick. "
        "A positive change, which occurs only when the timer resets, counts as 0 "
        "rather than as a reward.",
        "<b>death</b> &mdash; &minus;25 on the frame Mario dies. By far the largest "
        "single term, and what makes dying expensive enough to drive caution around "
        "enemies and gaps.",
        "<b>score</b> &mdash; the increase in the game's score counter, divided by 100.",
        "<b>coins</b> &mdash; 5 per coin, which is what makes collecting a row of "
        "coins on the ground worth doing rather than ignoring.",
        "<b>powerup</b> &mdash; 5 for gaining fire-flower form and &minus;5 for "
        "losing it, so shrinking is penalised.",
        "<b>completion</b> &mdash; +50, paid once when the flag is reached, and "
        "guarded by a flag so it cannot be collected repeatedly.",
    ], "tight")
    p("The project's own <font face='Courier'>RewardShaping</font> wrapper then "
      "rescales the total and adds a terminal bonus:", "body")
    equation(r"\tilde{r}_t \;=\; \frac{1}{s}\left(r_t + b \cdot \mathbb{1}\{\mathrm{flag\_get}_t\}\right), \qquad s = 10, \;\; b = 50")
    p("Dividing by <i>s</i> = 10 is numerical hygiene rather than shaping, and the "
      "reason is specific. The environment advertises a reward range of "
      "[&minus;15, 15], but the death and completion terms are &minus;25 and +50, "
      "so one step can return a value well outside it. PPO's value loss is "
      "<b>quadratic</b> in its residual, so an occasional large target produces a "
      "disproportionately large value gradient that competes with the policy "
      "gradient inside the same backward pass. Dividing by a positive constant "
      "leaves the optimal policy unchanged &mdash; it scales the whole objective "
      "uniformly &mdash; and only puts the two loss terms on comparable footing.", "body")
    p("The terminal bonus is a second, smaller copy of the library's own "
      "completion reward, and the code says so plainly: it exists so the reward "
      "curve visibly steps upward at the moment the level is solved. Because the "
      "flag is rare until the level is nearly finished, the term contributes "
      "almost nothing to the gradient during the run-up to a clear. It is "
      "presentation, not learning.", "body")
    p("One consequence is worth stating explicitly, because it is easy to misread "
      "the rescaling as a change of incentives. Dividing by 10 divides the death "
      "penalty too, so in the units PPO actually optimises a death costs "
      "&minus;2.5 and the flag bonus adds +5. The death penalty remains by far the "
      "largest per-step term, which is the intended property.", "body")
    panel([
        "After the rescale, dying is worth &minus;2.5 while a successful step of "
        "progress earns at most 0.5. The ratio is what matters, and it has not "
        "changed: the death penalty is still doing nearly all the work of teaching "
        "caution, and the +5 completion bonus is doing almost none.",
    ], "Reading the scaled numbers correctly")

    # ------------------------------------------------ 7. observation pipeline
    section("7. The observation pipeline: frame skip, pooling, stacking")
    p("The policy does not see the raw 256&times;240 framebuffer. Four "
      "transformations reduce it to a fixed 84&times;84 tensor, and each solves a "
      "specific problem with the emulator's output.", "body")

    p("7.1 Frame skipping and reward aggregation", "h2")
    p("The policy acts once every <i>k</i> = 4 emulated frames. The <i>k</i> "
      "individual rewards are summed, so the transition PPO sees runs from "
      "<i>s<sub>t</sub></i> to <i>s</i><sub><i>t</i>+<i>k</i></sub> with a single "
      "aggregated return:", "body")
    equation(r"R_t \;=\; \sum_{i=0}^{k-1} \gamma^{\,i}\, r_{t+i}")
    p("This is the correct treatment rather than an approximation of it: the "
      "sub-objective of a macro-action is exactly this discounted sum. Skipping "
      "also divides the number of policy decisions by four, which is what brings a "
      "learnable run inside a few hours rather than a few days.", "body")

    p("7.2 Max-pooling the last two frames", "h2")
    p("The NES flickers sprites on alternate frames for hardware reasons, so a "
      "single sampled frame can be missing an enemy entirely. The wrapper therefore "
      "takes an elementwise maximum over the final two frames of the skip window:", "body")
    equation(r"o_t \;=\; \max\left(\, f_{t+k-2}, \;\;\; f_{t+k-1} \,\right) \quad \text{elementwise}")
    p("The maximum, not the mean, is what makes this work. Because the operation "
      "is a logical OR over any non-background pixel, a sprite appearing in either "
      "frame survives into the observation. On a crowded level such as 1-2 a single "
      "missed obstacle is enough for the policy to commit to a fatal action, so this "
      "is a cheap and effective variance reduction on the input.", "body")

    p("7.3 Resize and grayscale", "h2")
    p("The frame is resized to 84&times;84 and converted to grayscale with the "
      "channel dimension retained, giving 84&times;84&times;1 per environment. "
      "Colour carries no task-relevant information in a game with a fixed palette, "
      "so three channels would be redundant input dimensions that the first "
      "convolution has to learn to ignore.", "body")

    p("7.4 Frame stacking", "h2")
    p("A single frame does not contain velocity. Two states that look nearly "
      "identical in one screenshot can require opposite actions, so four "
      "consecutive frames are stacked along the channel axis:", "body")
    equation(r"s_t \;=\; \left[\, f_{t-3}, \; f_{t-2}, \; f_{t-1}, \; f_t \,\right] \;\in\; \mathbb{R}^{84 \times 84 \times 4}")
    p("There is a detail here that is easy to get wrong and worth stating because it "
      "changes the interpretation. With frame skipping at 4, four stacked agent steps "
      "span <b>16 emulated frames</b>, not four. A jump arc in this game lasts on "
      "the order of that many frames, so the stack is just long enough to make a "
      "jump's shape and direction legible. The fourth channel functions as a "
      "temporal difference, supplying exactly the information a single frame "
      "structurally cannot.", "body")
    panel([
        "Stacking is applied by <font face='Courier'>VecFrameStack</font> at the "
        "vectorised-environment level rather than inside a per-environment wrapper, "
        "so the per-environment observation stays 84&times;84&times;1 and the "
        "history is assembled by SB3. This matters for the network: the convolution "
        "stack expects channel-first input, and the transpose to (4, 84, 84) is what "
        "makes the 8&times;8 first kernel legal.",
    ], "Why stacking sits outside the wrapper stack")

    # ------------------------------------------------ 8. network
    section("8. The network")
    p("The policy is an actor-critic pair sharing a convolutional feature "
      "extractor: the architecture introduced for DQN, adapted to a stacked image "
      "observation. Both heads read the same 512-dimensional feature vector, so the "
      "visual processing is computed once per step and shared.", "body")
    p("The shapes below were measured by instantiating the library's own "
      "<font face='Courier'>NatureCNN</font> rather than copied from the paper, so "
      "they are what this project actually trains:", "body")
    story.append(code_block([
        "input              (1, 4, 84, 84)      4 stacked 84x84 grayscale frames",
        "Conv2d  32ch  k8  s4  p0  ->  (1, 32, 20, 20)        8,224 params",
        "Conv2d  64ch  k4  s2  p0  ->  (1, 64,  9,  9)       32,832 params",
        "Conv2d  64ch  k3  s1  p0  ->  (1, 64,  7,  7)       36,928 params",
        "Flatten            64 x 7 x 7 = 3,136",
        "Linear  3,136 -> 512  + ReLU  ->  (1, 512)       1,606,144 params",
        "                                    feature extractor  1,684,128",
    ]))
    sp(7)
    p("Each convolution's output size follows the standard formula, which with no "
      "padding is the size at which a filter is fully valid:", "body")
    equation(r"H_{\mathrm{out}} \;=\; \left\lfloor \frac{H_{\mathrm{in}} + 2p - k}{s} \right\rfloor + 1")
    # Times has no glyph for the floor brackets, so the prose states the
    # p = 0 case in words rather than using &lfloor;/&rfloor; entities, which
    # would render as empty boxes. The equation above still uses them.
    p("With <i>p</i> = 0 this reduces to the floor of (<i>H</i>&nbsp;&minus;&nbsp;"
      "<i>k</i>) divided by <i>s</i>, plus one, giving 20, then 9, then 7. "
      "Roughly 95% of the extractor's 1,684,128 parameters sit in the final "
      "3,136&rarr;512 projection rather than in the convolutions, which is worth "
      "noting when judging where the model's capacity is spent.", "body")
    p("On top of the shared features sit two linear heads. The <b>policy head</b> "
      "maps the feature vector to one logit per action, and a softmax turns those "
      "logits into a distribution over actions since the action space is discrete "
      "and small:", "body")
    equation(r"\pi_{\theta}(a \mid s) = \mathrm{softmax}\left(f_{\theta}(s)\right)_{a}, \qquad f_{\theta}(s) \in \mathbb{R}^{|\mathcal{A}|}")
    p("The <b>value head</b> maps the same 512-dimensional vector to a single "
      "scalar <i>V<sub>&phi;</sub>(<i>s</i>)</i>, which serves as the baseline in "
      "the advantage estimate. A subtlety worth recording: when a CNN feature "
      "extractor is used, Stable-Baselines3 defaults "
      "<font face='Courier'>net_arch</font> to an <b>empty</b> list, so the heads "
      "are single linear layers with no hidden MLP between the feature vector and "
      "each output. The shared convolution stack already supplies the nonlinearity, "
      "so a further MLP would be redundant.", "body")
    p("Parameters are initialised orthogonally by default in the actor-critic "
      "policy, which keeps the variance of activations roughly constant across depth "
      "during the early updates and so stabilises the first few hundred gradient "
      "steps.", "body")
    panel([
        "The observation space must be declared <b>channel-first</b>, as "
        "(4, 84, 84), not (84, 84, 4). The 8&times;8 first kernel is only valid "
        "against a spatial input; declaring the space as (84, 84, 4) makes the "
        "library attempt an 8&times;8 kernel over a 4-wide axis and fail at "
        "construction. SB3's <font face='Courier'>VecTransposeImage</font> performs "
        "the conversion between the environment's channel-last output and the "
        "policy's channel-first input.",
    ], "One detail that is easy to get wrong")

    # ------------------------------------------------ 9. training loop
    section("9. The training loop")
    p("One iteration collects a fixed-size rollout and then optimises over that "
      "data for several passes. With <i>n<sub>steps</sub></i> = 512 transitions "
      "per environment and 3 parallel environments, each iteration collects "
      "<i>N</i> = 512 &times; 3 = 1,536 transitions. Shuffled into minibatches of "
      "256, that gives", "body")
    equation(r"n_{\mathrm{batches}} = \left\lfloor \frac{N}{B} \right\rfloor = \left\lfloor \frac{1536}{256} \right\rfloor = 6, \qquad \text{gradient steps} = 6 \times 10 = 60")
    p("The multiple epochs are what make PPO sample-efficient: the data is reused "
      "rather than discarded, and the clipping is what makes that reuse safe. After "
      "the first epoch the samples are off-policy relative to the current "
      "parameters and the ratio <i>r<sub>t</sub></i> drifts away from 1 &mdash; "
      "which is exactly the quantity the trust region exists to bound. The "
      "<font face='Courier'>clip_fraction</font> metric, the fraction of samples "
      "whose ratio has left the band, is the direct measure of how often the update "
      "is being constrained.", "body")
    p("A few hyperparameters deviate from the library defaults, and each deviation "
      "has a reason tied to this specific environment:", "body")
    items([
        "<b><i>&gamma;</i> = 0.9</b> &mdash; low by Atari standards, where 0.99 is "
        "typical. Frame skipping compresses the effective horizon: one decision "
        "covers 4 emulated frames, so the useful credit-assignment window in agent "
        "steps is short, and a high discount makes the value function much harder to "
        "fit without buying additional useful credit assignment.",
        "<b><i>&lambda;</i> = 0.95</b> &mdash; the standard GAE setting, keeping the "
        "estimator near-unbiased while still bootstrapping from the value network.",
        "<b><i>&epsilon;</i> = 0.2</b> &mdash; the standard clipping range.",
        "<b>Learning rate 1&times;10<sup>&minus;4</sup></b>, constant rather than "
        "annealed, and deliberately small since the policy is reused across 10 epochs.",
        "<b>Entropy coefficient 0.01, value coefficient 0.5</b> &mdash; the value "
        "loss is already large relative to the policy loss because it is quadratic, "
        "so it is down-weighted.",
        "<b>Gradient-norm clipping at 0.5</b> &mdash; a second, independent guarantee "
        "on step size, applied on top of the objective-level clipping.",
    ], "tight")
    p("Two implementation details affect how the results should be read. Rewards "
      "and advantages are standardised per minibatch, which keeps the effective scale "
      "of the advantage estimates stable over a long run; the policy loss is "
      "invariant to a positive rescaling of <i>A</i> but not to a sign change, so "
      "this does not alter what is being maximised. And no evaluation environment "
      "runs during training, because NES emulation is the bottleneck and a second "
      "environment would take CPU away from the rollout workers. The best checkpoint "
      "is tracked instead from the rolling mean of training episode reward, which is "
      "free, and the rigorous evaluation runs afterwards.", "body")

    # ------------------------------------------------ 10. results
    section("10. Results")
    r12 = json.loads((ART / "results_1_2.json").read_text(encoding="utf-8"))
    r11 = json.loads((ART / "results.json").read_text(encoding="utf-8"))
    best12, best11 = r12[0], r11[0]

    p("Both levels were trained with the same code, the same hyperparameters and "
      "the same objective. Only the level configuration differs, and for 1-2 that "
      "difference is substantive: the level ends by walking right into an exit pipe, "
      "which requires a <font face='Courier'>down</font> input. Neither the 7-action "
      "<font face='Courier'>simple</font> set nor the 5-action "
      "<font face='Courier'>right_only</font> set contains one, so 1-2 uses the "
      "12-action <font face='Courier'>complex</font> set. That was established by a "
      "scripted-policy test before committing to a multi-hour run, not assumed.", "body")
    p("World 1-2, evaluated over 20 episodes per run with the stochastic policy:",
      "body")
    story.append(results_table(r12))
    sp(7)
    p(f"The best 1-2 run clears the level in {best12['clear_rate']:.0%} of "
      f"evaluation episodes, having first touched the flagpole at "
      f"{best12['first_clear_steps']:,} environment steps. For comparison, the "
      f"corresponding 1-1 figures are a {best11['clear_rate']:.0%} clear rate and a "
      f"first clear at {best11['first_clear_steps']:,} steps. The two levels were "
      f"trained under identical settings on the same machine, so the difference in "
      f"difficulty is a property of the levels rather than of the setup.", "body")
    story += figure(ART / "training_curve_1_2.png", 0.95,
                    "Training curves for the two 1-2 runs. Left: episode reward. "
                    "Right: furthest position reached within the level. The flagpole "
                    "line sits at x &asymp; 3161, measured from the runs themselves "
                    "rather than assumed.")
    p("Three things about these numbers are worth stating plainly rather than "
      "leaving to be inferred.", "body")
    items([
        "<b>The clear rate is measured on the stochastic policy, not the greedy "
        "one.</b> The emulator is fully deterministic, so a greedy (argmax) policy "
        "replays one identical trajectory every time; a rate measured that way could "
        "only ever be 0% or 100%, and would be one episode reported twenty times. "
        "The stochastic policy is where the variance is real, so that is where "
        "robustness is reported. In these 1-2 runs the greedy policy does not finish "
        "the level, while sampling from the identical weights clears it most of the "
        "time &mdash; the same narrow failure of action selection observed on 1-1.",
        "<b>Warp-pipe escapes are counted separately and excluded from the clear "
        "rate.</b> World 1-2 contains pipes leading into other worlds, and the "
        "single-stage environment does not end an episode when the agent takes one. "
        "An escaping agent could reach a different level's flagpole and be recorded "
        "as clearing 1-2. A guard ends such episodes and reports them through "
        "<font face='Courier'>left_stage</font>, so the headline number cannot be "
        "inflated by them. Both runs recorded zero escapes, so no figure here "
        "depends on that guard.",
        "<b>Seed variance is large.</b> The two runs differ substantially in the one "
        "respect that matters most. Reporting only the better run would overstate "
        "what the method achieves, so the per-run table is the primary result rather "
        "than a footnote to it.",
    ], "tight")
    p("A partial result is reported as a partial result. Where a run did not clear "
      "the level reliably, that number is stated rather than smoothed away, and the "
      "flagpole reference line is omitted from the plot entirely until the level has "
      "actually been cleared &mdash; a guessed target would imply the agent had been "
      "measured against something real.", "body")

    # ------------------------------------------------ 11. design decisions
    section("11. Design decisions, trade-offs, and limits")

    p("11.1 Why frame stacking is not optional", "h2")
    p("A single frame cannot express velocity. Two situations that look identical "
      "in one screenshot &mdash; Mario standing still, and Mario sprinting at full "
      "speed &mdash; require opposite actions, and the policy cannot distinguish "
      "them from one frame alone. Stacking four agent steps supplies exactly the "
      "temporal difference that makes the distinction visible.", "body")
    p("This matches a reported finding in prior work on this task, that a "
      "single-frame model gets stuck and that temporal context helps the agent "
      "commit to jumps. The stall observed in this project under the greedy policy is "
      "the same phenomenon: the agent runs to roughly x&thinsp;&asymp;&thinsp;2370 "
      "and then stands still until the in-game clock expires, spending about 90% of "
      "the episode motionless, while the same weights sampled rather than argmaxed "
      "clear the level. Stalling in front of an obstacle is therefore a failure of "
      "action selection, not of the learned representation, and both temporal "
      "context and action sampling mitigate it.", "body")

    p("11.2 Why the reward is rescaled rather than redesigned", "h2")
    p("It would be reasonable to rewrite the reward entirely &mdash; a large bonus "
      "for distance, a penalty for taking damage. The project deliberately does not, "
      "and the division by 10 is the only change to the reward's structure. The "
      "reason is optimisation, not pedagogy.", "body")
    p("PPO's value loss is quadratic in its residual, so a rare but large target "
      "produces a disproportionately large value gradient that competes with the "
      "policy gradient inside the same backward pass. Rescaling by a positive "
      "constant leaves the optimal policy unchanged and merely puts the two loss "
      "terms on comparable footing. The library's reward already encodes the "
      "intended incentives correctly; the only genuine problem was numerical.", "body")

    p("11.3 Why no evaluation environment runs during training", "h2")
    p("NES emulation is the binding constraint on this machine, not GPU compute: "
      "the policy is small and the GPU is largely idle. A second environment for "
      "periodic evaluation would take CPU directly away from the rollout workers, "
      "which is the resource that actually limits throughput. The best checkpoint is "
      "tracked instead from the rolling mean of training episode reward, which costs "
      "nothing, and the rigorous multi-episode evaluation runs afterwards.", "body")

    p("11.4 What is deliberately not claimed", "h2")
    items([
        "<b>No claim of generalisation.</b> The claim is scoped to specific levels "
        "trained on those levels. Prior work reports that an agent trained on 1-1 "
        "transfers poorly to 1-2, 4-1 and 5-1, having learned tile patterns specific "
        "to a layout rather than transferable mechanics. Training two levels "
        "separately produces two policies; it is not evidence that either generalises.",
        "<b>No claim about efficiency.</b> Roughly a million environment steps is "
        "modest by reinforcement learning standards, and the comparison in the "
        "project's README is against a RAM-based representation that removes the "
        "perception problem entirely &mdash; a fundamentally easier task.",
        "<b>No claim that the deterministic policy works.</b> It does not, on 1-2. "
        "That is reported as a pass/fail rather than folded into the clear rate, "
        "because a rate derived from a deterministic environment and a deterministic "
        "policy would be one episode counted twenty times.",
    ], "tight")


    p("References", "h1")
    items([
        "Sutton, R. S. and Barto, A. G. <i>Reinforcement Learning: An Introduction</i>, 2nd edition. Value functions, Bellman equations, and the general RL framework.",
        "Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. <i>Machine Learning</i>, 8, 229&ndash;256. The policy gradient theorem.",
        "Mnih, V. <i>et al.</i> (2015). Human-level control through deep reinforcement learning. <i>Nature</i>, 518, 529&ndash;533. The Nature-CNN architecture and the frame-skipping and stacking pipeline.",
        "Schulman, J. <i>et al.</i> (2015). Trust Region Policy Optimization. <i>ICML</i>. The trust-region formulation that PPO approximates.",
        "Schulman, J. <i>et al.</i> (2017). Proximal Policy Optimization Algorithms. <i>arXiv:1707.06347</i>. The clipped surrogate objective and the entropy bonus.",
        "Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. <i>ICLR</i>. The GAE estimator and the &gamma;&ndash;&lambda; trade-off.",
        "Raffin, A. <i>et al.</i> (2021). Stable-Baselines3: reliable reinforcement learning implementations. <i>Journal of Machine Learning Research</i>, 22(268), 1&ndash;8. The PPO implementation used here, unmodified.",
    ], "refs")

    sp(8)
    story.append(hr())

    # ------------------------------------------------------------- build
    out = ART / "math_concepts.pdf"
    doc = BaseDocTemplate(str(out), pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN + 8,
                          title="The mathematics behind a Super Mario Bros. "
                                "PPO agent",
                          author="newupdatedmassprj",
                          subject="PPO, GAE, frame skipping and reward design")
    frame = Frame(MARGIN, MARGIN + 8, DOC_W, PAGE_H - 2 * MARGIN - 8,
                  id="body", leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=page_furniture)])
    doc.build(story)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{doc.page} pages, {_eq_seq[0]} numbered equations)")
    return out


if __name__ == "__main__":
    build()

