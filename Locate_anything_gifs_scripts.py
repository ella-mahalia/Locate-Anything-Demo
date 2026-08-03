"""
Locate_anything_gifs_scripts.py

Standalone script that regenerates four animated diagrams for
NVIDIA's LocateAnything-3B (https://huggingface.co/nvidia/LocateAnything-3B):

1. hybrid_mode_flow_v3.gif        - Hybrid decoding mode decision flow
2. locateanything_architecture_v5.gif - Full architecture pipeline (encoder -> projector -> decoder -> PBD)
3. multitask_grounding_v4.gif      - Multi-task grounding demo (object detection / GUI / OCR)
4. vision_tokenization_v3.gif      - MoonViT native-resolution tokenization walkthrough

Requirements:
    pip install matplotlib pillow numpy

Run:
    python Locate_anything_gifs_scripts.py

Notes:
- Diagram 3 (multitask_grounding) downloads three reference images
  (street scene, mobile login screen, scanned document page) into an
  "assets" folder on first run. An internet connection is required for
  that step only; diagrams 1, 2, and 4 use no external assets besides
  a synthetic image for diagram 4's tokenization example is generated
  in-script from the same downloaded street photo.
"""

import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

OUTPUT_DIR = "output"
ASSET_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)


def draw_box(ax, x, y, w, h, text, color, fontsize=9):
    """Draw a rounded, semi-transparent labeled box on the given axes."""
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.25
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color="white", weight="bold")
    return rect


def download_assets():
    """Download reference images used by the multitask grounding diagram."""
    urls = {
        "street.jpg": "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/7c24dd68aa7e2102248c2feb30de3eeffebc6ba6.jpg",
        "login.jpg": "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/4afb29e812e19a63e84469b430b7689ba3761ccd.jpg",
        "doc.jpg": "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/2aa5a81779116ecfddfa50fc07e78c39c1d4d5de.jpg",
    }
    for name, url in urls.items():
        path = os.path.join(ASSET_DIR, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)


# ----------------------------------------------------------------------
# 1. Hybrid Decoding Mode: Decision Flow (v3)
# ----------------------------------------------------------------------
def build_hybrid_mode_flow():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 8); ax.axis("off")
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")

    ax.text(5.5, 7.6, "LocateAnything Hybrid Decoding Mode: Decision Flow",
            ha="center", color="white", fontsize=13, weight="bold")
    status = ax.text(5.5, 7.1, "", color="#ffffff", fontsize=12, weight="bold", ha="center")

    draw_box(ax, 4.2, 5.9, 2.6, 1.0, "Start: New Object\nQuery", "#00d4ff")
    draw_box(ax, 3.9, 4.2, 3.2, 1.0, "Fast Mode (PBD)\nPredict full box in\n1 parallel step", "#06d6a0")
    diamond = patches.FancyBboxPatch((3.6, 2.4), 3.8, 1.1, boxstyle="round,pad=0.02",
                                      linewidth=1.5, edgecolor="#ffd166", facecolor="#ffd166", alpha=0.25)
    ax.add_patch(diamond)
    ax.text(5.5, 2.95, "Format OK &\nlow spatial ambiguity?", ha="center", va="center",
            color="white", fontsize=9, weight="bold")
    draw_box(ax, 0.3, 0.4, 3.2, 1.0, "Accept Box\n(Fast path)", "#8ac926")
    draw_box(ax, 6.8, 0.4, 3.6, 1.0, "Slow Mode (NTP)\nFallback token-by-token\ndecoding", "#ff595e")

    ax.annotate("", xy=(5.5, 5.2), xytext=(5.5, 5.9), arrowprops=dict(arrowstyle="->", color="white"))
    ax.annotate("", xy=(5.5, 3.5), xytext=(5.5, 4.2), arrowprops=dict(arrowstyle="->", color="white"))
    ax.annotate("", xy=(1.9, 1.4), xytext=(4.4, 2.6), arrowprops=dict(arrowstyle="->", color="#8ac926"))
    ax.annotate("", xy=(8.2, 1.4), xytext=(6.6, 2.6), arrowprops=dict(arrowstyle="->", color="#ff595e"))
    ax.text(2.6, 2.05, "Yes", color="#8ac926", fontsize=9, weight="bold")
    ax.text(7.6, 2.05, "No", color="#ff595e", fontsize=9, weight="bold")

    pulse = ax.scatter([], [], s=220, color="#ffffff", zorder=6)
    path_easy = [(5.5, 6.4), (5.5, 5.2), (5.5, 4.7), (5.5, 3.5), (5.5, 2.95), (3.0, 1.7), (1.9, 0.9)]
    path_hard = [(5.5, 6.4), (5.5, 5.2), (5.5, 4.7), (5.5, 3.5), (5.5, 2.95), (7.5, 1.7), (8.6, 0.9)]

    def update(frame):
        cycle = frame % 40
        use_easy = (frame // 40) % 2 == 0
        path = path_easy if use_easy else path_hard
        t = cycle / 40
        seg = t * (len(path) - 1)
        i = min(int(seg), len(path) - 2)
        frac = seg - i
        x = path[i][0] + (path[i + 1][0] - path[i][0]) * frac
        y = path[i][1] + (path[i + 1][1] - path[i][1]) * frac
        pulse.set_offsets([[x, y]])
        status.set_text("Simple, unambiguous object -> Fast path" if use_easy
                         else "Cluttered / ambiguous scene -> Slow fallback")
        status.set_color("#8ac926" if use_easy else "#ff595e")
        return pulse, status

    anim = FuncAnimation(fig, update, frames=80, interval=60)
    anim.save(os.path.join(OUTPUT_DIR, "hybrid_mode_flow_v3.gif"), writer=PillowWriter(fps=9))
    plt.close(fig)


# ----------------------------------------------------------------------
# 2. Full Architecture Pipeline (v5) - synchronized pulse + step-by-step decoding
# ----------------------------------------------------------------------
def build_architecture_pipeline():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.5); ax.axis("off")
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")

    draw_box(ax, 0.3, 5.5, 1.8, 1.3, "Input Image\n+ Text Query", "#00d4ff")
    draw_box(ax, 2.6, 5.5, 2.0, 1.3, "MoonViT-SO-400M\nVision Encoder\n(native-res tokens)", "#ff9f1c")
    draw_box(ax, 5.1, 5.5, 1.8, 1.3, "MLP\nProjector", "#8ac926")
    draw_box(ax, 7.4, 5.5, 2.2, 1.3, "Qwen2.5\nLanguage Decoder", "#ff595e")
    draw_box(ax, 10.1, 5.5, 2.4, 1.3, "Parallel Box\nDecoding Head\n(PBD)", "#bc6ff1")
    for x0, x1 in [(2.1, 2.6), (4.6, 5.1), (6.9, 7.4), (9.6, 10.1)]:
        ax.annotate("", xy=(x1, 6.15), xytext=(x0, 6.15), arrowprops=dict(arrowstyle="->", color="white"))
    ax.text(6.5, 7.1, "LocateAnything-3B Architecture: Vision Encoder -> Projector -> LM -> Parallel Box Decoding",
            ha="center", color="white", fontsize=12, weight="bold")

    ax.text(1.5, 4.6, "Slow Mode (NTP): sequential token-by-token", color="#ffd166", fontsize=10, weight="bold")
    ax.text(1.5, 2.3, "Fast Mode (PBD): atomic block, one parallel step", color="#06d6a0", fontsize=10, weight="bold")

    labels = ["label", "x1", "y1", "x2", "y2", "pad"]
    ntp_cells = [draw_box(ax, 1.5 + i * 1.6, 3.9, 1.4, 0.55, lab, "#3a3a3a", 8) for i, lab in enumerate(labels)]
    pbd_cells = [draw_box(ax, 1.5 + i * 1.6, 1.6, 1.4, 0.55, lab, "#3a3a3a", 8) for i, lab in enumerate(labels)]

    pulse_dot = ax.scatter([], [], s=220, color="#00d4ff", zorder=6)
    stages_x = [1.2, 3.6, 6.0, 8.5, 11.3]
    n_stages = len(stages_x)

    ntp_dot = ax.scatter([], [], s=140, color="#ffd166", zorder=6)
    pbd_dots = ax.scatter([], [], s=140, color="#06d6a0", zorder=6)

    status_text = ax.text(9.5, 0.6, "", color="white", fontsize=10, ha="left")

    STEP_FRAMES = 4
    NTP_CYCLE = 6 * STEP_FRAMES  # frames for one full sequential 6-token pass
    n_frames = NTP_CYCLE * n_stages  # top pulse advances once per completed NTP cycle

    def update(frame):
        # Top pipeline pulse only advances to the next stage after a full
        # NTP cycle (6 sequential steps) has completed underneath it.
        cycle_num = (frame // NTP_CYCLE) % n_stages
        within_cycle = frame % NTP_CYCLE
        step = within_cycle // STEP_FRAMES

        pulse_dot.set_offsets([[stages_x[cycle_num], 6.15]])

        cx = 1.5 + step * 1.6 + 0.7
        ntp_dot.set_offsets([[cx, 4.17]])
        for j, c in enumerate(ntp_cells):
            c.set_alpha(0.55 if j < step else (0.8 if j == step else 0.25))
            c.set_edgecolor("#ffd166" if j == step else "#3a3a3a")

        if within_cycle < STEP_FRAMES * 2:
            pbd_dots.set_offsets([[1.5 + i * 1.6 + 0.7, 1.87] for i in range(6)])
            pbd_dots.set_alpha(0.9)
            for c in pbd_cells:
                c.set_alpha(0.65); c.set_edgecolor("#06d6a0")
        else:
            pbd_dots.set_offsets(np.empty((0, 2)))
            for c in pbd_cells:
                c.set_alpha(0.25); c.set_edgecolor("#3a3a3a")

        status_text.set_text(
            "PBD completes a full 6-token box in 1 parallel step vs\n"
            "NTP's 6 sequential steps (~10x higher throughput)"
        )
        return ntp_cells + pbd_cells + [pulse_dot, ntp_dot, pbd_dots, status_text]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=90)
    anim.save(os.path.join(OUTPUT_DIR, "locateanything_architecture_v5.gif"), writer=PillowWriter(fps=11))
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. Multi-Task Grounding Demo (v4) - real images, boxes on real objects
# ----------------------------------------------------------------------
def build_multitask_grounding():
    download_assets()
    img_street = mpimg.imread(os.path.join(ASSET_DIR, "street.jpg"))
    img_login = mpimg.imread(os.path.join(ASSET_DIR, "login.jpg"))
    img_doc = mpimg.imread(os.path.join(ASSET_DIR, "doc.jpg"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.patch.set_facecolor("#0f1117")
    task_names = ["Object Detection", "GUI Grounding", "OCR / Doc Layout"]
    task_colors = ["#00d4ff", "#ff9f1c", "#8ac926"]
    imgs = [img_street, img_login, img_doc]

    # Boxes tuned to sit on visible objects (x, y, w, h) in axis-fraction coords
    box_sets = [
        [(0.05, 0.42, 0.22, 0.28), (0.38, 0.40, 0.20, 0.30), (0.60, 0.15, 0.30, 0.30)],  # street: cars/pedestrians
        [(0.30, 0.68, 0.40, 0.16), (0.15, 0.38, 0.70, 0.24), (0.25, 0.20, 0.50, 0.10)],  # login: logo/fields/button
        [(0.08, 0.82, 0.55, 0.08), (0.08, 0.68, 0.75, 0.08), (0.08, 0.15, 0.85, 0.45)],  # doc: title/heading/body
    ]

    rect_artists = []
    for ax, name, color, boxes, im in zip(axes, task_names, task_colors, box_sets, imgs):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(im, extent=[0, 1, 0, 1], aspect="auto", zorder=1)
        ax.set_title(name, color=color, fontsize=11, weight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        artists = []
        for (x, y, w, h) in boxes:
            r = patches.Rectangle((x, y), w, h, linewidth=2.5, edgecolor=color, facecolor="none", alpha=0, zorder=5)
            ax.add_patch(r)
            artists.append(r)
        rect_artists.append(artists)
    fig.suptitle("LocateAnything: One Model, Multiple Grounding Tasks", color="white", fontsize=13, weight="bold")

    def update(frame):
        step = frame % 33
        idx = step // 11
        for artists in rect_artists:
            for j, r in enumerate(artists):
                if j < idx:
                    r.set_alpha(0.95)
                elif j == idx:
                    r.set_alpha((step % 11) / 11)
                else:
                    r.set_alpha(0)
        return [r for artists in rect_artists for r in artists]

    anim = FuncAnimation(fig, update, frames=33, interval=180)
    anim.save(os.path.join(OUTPUT_DIR, "multitask_grounding_v4.gif"), writer=PillowWriter(fps=5))
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. Native-Resolution Vision Tokenization (v3) - full pipeline explainer
# ----------------------------------------------------------------------
def build_vision_tokenization():
    download_assets()
    img = mpimg.imread(os.path.join(ASSET_DIR, "street.jpg"))

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
    ax.text(6.5, 7.6, "MoonViT: Native-Resolution Image Tokenization",
            ha="center", color="white", fontsize=13, weight="bold")
    ax.text(6.5, 7.15,
            "Unlike fixed-resolution ViTs, MoonViT preserves the image's original aspect ratio,\n"
            "avoiding distortion from resizing/cropping before tokenization",
            ha="center", color="#aaaaaa", fontsize=9)

    # Stage 1: raw image
    ax.text(2.1, 6.5, "1. Raw Input Image", color="#00d4ff", ha="center", fontsize=10, weight="bold")
    ax.imshow(img, extent=[0.4, 3.8, 2.6, 6.1], aspect="auto", zorder=1)
    ax.add_patch(patches.Rectangle((0.4, 2.6), 3.4, 3.5, linewidth=2, edgecolor="#00d4ff", facecolor="none", zorder=2))

    # Stage 2: patch grid overlay
    nrows, ncols = 7, 7
    patch_w, patch_h = 3.4 / ncols, 3.5 / nrows
    grid_patches = []
    for r in range(nrows):
        for c in range(ncols):
            p = patches.Rectangle((0.4 + c * patch_w, 2.6 + r * patch_h), patch_w, patch_h,
                                   linewidth=0.7, edgecolor="#ffffff", facecolor="none", alpha=0.35, zorder=3)
            ax.add_patch(p)
            grid_patches.append(p)
    ax.text(2.1, 2.2, "2. Split into variable-count\nnative-resolution patches",
            color="#ff9f1c", ha="center", fontsize=9, weight="bold")

    ax.annotate("", xy=(4.6, 4.35), xytext=(3.9, 4.35), arrowprops=dict(arrowstyle="->", color="white", lw=2))

    # Stage 3: token sequence
    tok_x0, tok_y0 = 4.9, 3.0
    token_boxes = []
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        tb = patches.FancyBboxPatch((tok_x0 + c * 0.34, tok_y0 + (nrows - 1 - r) * 0.34), 0.28, 0.28,
                                     boxstyle="round,pad=0.01", linewidth=0.6,
                                     edgecolor="#ff9f1c", facecolor="#ff9f1c", alpha=0.15, zorder=4)
        ax.add_patch(tb)
        token_boxes.append(tb)
    ax.text(7.0, 6.5, "3. Flattened token sequence\n(each token = one image patch embedding)",
            color="#ff9f1c", ha="center", fontsize=9, weight="bold")

    ax.annotate("", xy=(8.0, 4.35), xytext=(7.4, 4.35), arrowprops=dict(arrowstyle="->", color="white", lw=2))

    # Stage 4: transformer encoder block
    draw_box(ax, 8.3, 2.8, 2.0, 3.0, "MoonViT\nTransformer\nEncoder\nLayers", "#ff595e", fontsize=10)
    ax.annotate("", xy=(11.0, 4.35), xytext=(10.3, 4.35), arrowprops=dict(arrowstyle="->", color="white", lw=2))

    # Stage 5: output feature vector
    draw_box(ax, 11.0, 3.4, 1.7, 1.8, "Contextualized\nVisual Features\n-> LM", "#8ac926", fontsize=8)
    ax.text(11.85, 5.5, "4. Output", color="#8ac926", ha="center", fontsize=9, weight="bold")

    info_text = ax.text(6.5, 1.3, "", color="white", fontsize=10, ha="center", weight="bold")

    def update(frame):
        idx = frame % (nrows * ncols)
        for j, gp in enumerate(grid_patches):
            gp.set_edgecolor("#00d4ff" if j == idx else "#ffffff")
            gp.set_alpha(0.9 if j == idx else 0.3)
            gp.set_linewidth(2.0 if j == idx else 0.7)
        for j, tb in enumerate(token_boxes):
            if j <= idx:
                tb.set_facecolor("#ff9f1c"); tb.set_alpha(0.85)
            else:
                tb.set_facecolor("#333333"); tb.set_alpha(0.3)
        info_text.set_text(
            f"Patch {idx + 1}/{nrows * ncols} converted to a visual token "
            "— no resizing distortion, resolution-independent"
        )
        return grid_patches + token_boxes + [info_text]

    anim = FuncAnimation(fig, update, frames=nrows * ncols, interval=200)
    anim.save(os.path.join(OUTPUT_DIR, "vision_tokenization_v3.gif"), writer=PillowWriter(fps=5))
    plt.close(fig)


if __name__ == "__main__":
    print("Building hybrid_mode_flow_v3.gif ...")
    build_hybrid_mode_flow()
    print("Building locateanything_architecture_v5.gif ...")
    build_architecture_pipeline()
    print("Building multitask_grounding_v4.gif ...")
    build_multitask_grounding()
    print("Building vision_tokenization_v3.gif ...")
    build_vision_tokenization()
    print("All diagrams saved to the 'output' folder.")
