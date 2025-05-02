import cv2
import numpy as np
import os
import glob


def order_points(pts):
    # sort the coordinates based on their x+y sum and difference
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top‑left
    rect[2] = pts[np.argmax(s)]  # bottom‑right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top‑right
    rect[3] = pts[np.argmax(diff)]  # bottom‑left
    return rect


class CardDetector:
    """Detects playing cards on a dark table and returns their names."""

    def __init__(self, templates_path: str):
        """
        Parameters
        ----------
        templates_path : str
            Directory containing two subfolders:
            - `ranks` with one‑bit PNGs named `ace.png`, `2.png`, …, `king.png`
            - `suits` with `hearts.png`, `spades.png`, etc.
        """
        self.rank_templates: dict[str, np.ndarray] = {}
        self.suit_templates: dict[str, np.ndarray] = {}
        self._load_templates(templates_path)

    # ‑‑‑ template handling ‑‑‑
    def _load_templates(self, base: str) -> None:
        for f in glob.glob(os.path.join(base, "ranks", "*.png")):
            name = os.path.splitext(os.path.basename(f))[0]
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            self.rank_templates[name] = self._prep_template(img)
        for f in glob.glob(os.path.join(base, "suits", "*.png")):
            name = os.path.splitext(os.path.basename(f))[0]
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            self.suit_templates[name] = self._prep_template(img)

    @staticmethod
    def _prep_template(img: np.ndarray) -> np.ndarray:
        """Resize and edge‑filter the template for faster correlation."""
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return cv2.Canny(img, 50, 150)

    # ‑‑‑ main public API ‑‑‑
    def process_frame(self, frame_bgr: np.ndarray) -> list[tuple[np.ndarray, str]]:
        """Detect cards in *frame_bgr* and return list of (warped_card, name)."""
        cards = []
        for warp in self._detect_cards(frame_bgr):
            name = self._classify_card(warp)
            cards.append((warp, name))
        return cards

    # ‑‑‑ step 1: locate quadrilateral blobs on the dark background ‑‑‑
    def _detect_cards(self, frame: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cards = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            area = cv2.contourArea(approx)
            if len(approx) == 4 and area > 12_000:  # filter noise
                warp = self._warp_card(frame, approx.reshape(4, 2))
                cards.append(warp)
        return cards

    # ‑‑‑ perspective transform to an upright 200×300 image ‑‑‑
    @staticmethod
    def _warp_card(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = order_points(pts)
        width = int(max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0])))
        height = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3])))
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(image, M, (width, height))
        return warp

    # ‑‑‑ step 2: read rank + suit from the corner patch ‑‑‑
    def _classify_card(self, card_bgr: np.ndarray) -> str:
        h, w = card_bgr.shape[:2]
        corner = card_bgr[0 : int(0.28 * h), 0 : int(0.28 * w)]
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        # rough crop coordinates tuned on standard poker size
        rank_roi = thr[8:70, 8:50]
        suit_roi = thr[70:140, 8:50]
        rank = self._best_template_match(rank_roi, self.rank_templates)
        suit = self._best_template_match(suit_roi, self.suit_templates)
        return f"{rank.capitalize()} of {suit.capitalize()}" if rank and suit else "Unknown"

    # ‑‑‑ cross‑correlate against templates ‑‑‑
    @staticmethod
    def _best_template_match(roi: np.ndarray, templates: dict[str, np.ndarray]) -> str | None:
        best_name = None
        best_score = -np.inf
        for name, tmpl in templates.items():
            if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
                continue
            res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
            score = res.max()
            if score > best_score:
                best_score = score
                best_name = name
        return best_name if best_score > 0.55 else None  # empirical threshold


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect playing cards on a table.")
    parser.add_argument("--templates", required=True, help="Path to rank/suit template images")
    parser.add_argument("--image", help="Single image to test")
    args = parser.parse_args()

    detector = CardDetector(args.templates)

    if args.image:
        img = cv2.imread(args.image)
        results = detector.process_frame(img)
        for warp, name in results:
            cv2.putText(warp, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(name, warp)
        cv2.waitKey(0)
    else:
        # Demo on webcam
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            for warp, name in detector.process_frame(frame):
                cv2.putText(warp, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                frame = cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.imshow("Card Detector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
