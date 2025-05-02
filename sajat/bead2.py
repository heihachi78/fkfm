import cv2
import numpy as np
import sys
import os
from pathlib import Path

# --- geometric thresholds ----------------------------------------------------
MIN_AREA_RATIO        = 0.025     # minimum contour area (% of whole image)
ASPECT_LOW, ASPECT_HIGH = 1.25, 1.7
RECT_FILL_RATIO       = 0.65      # contour must fill ≥ 65 % of its minAreaRect
RECT_FILL_RATIO_BORDER = 0.40
PAD                   = 10
# -----------------------------------------------------------------------------

def order_points(pts):
    """Return points in TL, TR, BR, BL order, padded by –PAD pixels."""
    pts = np.array(pts, dtype="float32")
    s, d = pts.sum(1), np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0], ordered[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    ordered[1], ordered[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return ordered - PAD                             # shift back after padding

def find_card_quads(img):
    wb        = cv2.xphoto.createLearningBasedWB()
    balanced  = wb.balanceWhite(img)

    gray      = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
    padded    = cv2.copyMakeBorder(gray, PAD, PAD, PAD, PAD,
                                   cv2.BORDER_CONSTANT, value=0)

    edges     = cv2.Canny(cv2.GaussianBlur(padded, (7, 7), 0), 10, 200)

    cnts,_    = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    h, w      = img.shape[:2]
    img_area  = h * w
    quads     = []

    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA_RATIO*img_area:
            continue
        rect      = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ = rect
        if rw == 0 or rh == 0:
            continue
        aspect    = max(rw, rh) / min(rw, rh)
        if not ASPECT_LOW <= aspect <= ASPECT_HIGH:
            continue
        if cv2.contourArea(c) / (rw*rh) < RECT_FILL_RATIO:
            continue
        quads.append(order_points(cv2.boxPoints(rect)))
    return quads



def load_reference_sift(ref_dir: str, verbose: bool = True):
    sift = cv2.SIFT_create()

    ref_data = []
    ref_dir  = Path(ref_dir)

    if not ref_dir.exists():
        raise FileNotFoundError(f"The reference folder '{ref_dir}' does not exist.")

    for file in ref_dir.rglob("*"):
        if not file.is_file():
            continue
        img = cv2.imread(str(file))

        if img is None:
            if verbose:
                print(f"Skipped non-image file: {file}")
            continue

        wb        = cv2.xphoto.createLearningBasedWB()
        balanced  = wb.balanceWhite(img)

        kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]], dtype=np.float32)

        sharpened = cv2.filter2D(balanced, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)

        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        ref_data.append((file, kp, des))

        if verbose:
            print(f"{file.name:>25s}: {len(kp):4d} key-points")

    if verbose:
        print(f"\nLoaded {len(ref_data)} reference image(s).")
    return ref_data


def sift_on_quads(gray, quads):
    sift          = cv2.SIFT_create()
    per_card_data = []                                 # [(keypoints, descriptors), …]

    for quad in quads:
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(mask, [quad.astype(int)], 255)    # white where we accept kp
        kp, des = sift.detectAndCompute(gray, mask)
        per_card_data.append((kp, des))
    return per_card_data


def simple_bf_match(reference_db, card_kp_and_desc,
                    ratio: float = 0.75) -> list[tuple[str | None, int]]:
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    results = []

    for kp_card, des_card in card_kp_and_desc:
        best_ref, best_score = None, 0

        for ref_path, kp_ref, des_ref in reference_db:
            # k-NN match (k=2) and Lowe ratio filter
            knn      = bf.knnMatch(des_card, des_ref, k=2)
            good     = [m for m, n in knn if m.distance < ratio * n.distance]
            if len(good) > best_score:
                best_ref, best_score = ref_path, len(good)

        results.append((best_ref, best_score))

    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("használat: python bead2.py tesztkep.jpeg")
        sys.exit(1)

    img  = cv2.imread(sys.argv[1])
    if img is None:
        print("Failed to open image.")
        sys.exit(1)

    reference_db  = load_reference_sift("references")

    quads            = find_card_quads(img)
    gray             = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    card_kp_and_desc = sift_on_quads(gray, quads)

    for quad, (kp, _) in zip(quads, card_kp_and_desc):
        cv2.polylines(img, [quad.astype(int)], True, (0,255,0), 2)
        for point in kp:
            x, y = map(int, point.pt)
            cv2.circle(img, (x, y), 2, (255,0,0), -1)

    matches = simple_bf_match(reference_db, card_kp_and_desc)

    for i, (ref_path, score) in enumerate(matches):
        if ref_path is None:
            print(f"Card {i}: no reasonable match")
        else:
            print(f"Card {i}: best match → {ref_path.name}  ({score} good matches)")

    print(f"Detected {len(quads)} card(s) and {sum(len(k) for k,_ in card_kp_and_desc)} SIFT key-points.")
    cv2.imshow("Card detection with SIFT", img)
    cv2.waitKey(0)
