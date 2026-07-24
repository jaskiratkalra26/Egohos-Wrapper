import cv2
import numpy as np

class EgoHOSPostProcessor:
    def __init__(
        self,
        max_distance_px: int = 150,
        morph_kernel_size: int = 5,
        min_blob_area_px: int = 200,
        use_connected_components: bool = True
    ):
        """
        Initializes the post-processor with customizable parameters to fix mask leakage.
        
        Args:
            max_distance_px: Maximum allowed distance from the hand mask for an object mask pixel.
            morph_kernel_size: Kernel size for morphological operations (erosion to break bridges).
            min_blob_area_px: Minimum pixel area to keep a connected component.
            use_connected_components: If True, uses connected components to ensure object blobs physically touch the hand.
        """
        self.max_distance_px = max_distance_px
        self.morph_kernel_size = morph_kernel_size
        self.min_blob_area_px = min_blob_area_px
        self.use_connected_components = use_connected_components

    def geofence_mask(self, hand_mask: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        """
        Erases object mask pixels that are too far from the hand mask.
        """
        if not np.any(hand_mask) or not np.any(object_mask):
            return object_mask

        # Find the center of the hand
        ys, xs = np.where(hand_mask > 0)
        hand_center_y = int(np.mean(ys))
        hand_center_x = int(np.mean(xs))

        # Create a distance map
        h, w = object_mask.shape
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - hand_center_x)**2 + (Y - hand_center_y)**2)

        # Create a valid region mask
        valid_region = dist_from_center <= self.max_distance_px

        # Apply geofence
        cleaned_mask = object_mask.copy()
        cleaned_mask[~valid_region] = 0

        return cleaned_mask

    def filter_connected_components(self, hand_mask: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        """
        Ensures the object mask is physically touching the hand mask by finding
        connected components and removing isolated blobs.
        """
        if not np.any(object_mask):
            return object_mask

        # Erode to break thin "leakage bridges"
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        eroded_obj = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded_obj, connectivity=8)
        
        cleaned_eroded_obj = np.zeros_like(eroded_obj)
        
        # Check which blobs overlap or touch the dilated hand mask
        dilated_hand = cv2.dilate(hand_mask.astype(np.uint8), kernel, iterations=3)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_blob_area_px:
                continue

            # Create a mask for just this blob
            blob_mask = (labels == i).astype(np.uint8)

            # Check if this blob intersects with the dilated hand mask
            overlap = cv2.bitwise_and(blob_mask, dilated_hand)
            if np.any(overlap):
                # Valid blob! Add it back.
                cleaned_eroded_obj = cv2.bitwise_or(cleaned_eroded_obj, blob_mask)

        # Dilate back to original size
        cleaned_mask = cv2.dilate(cleaned_eroded_obj, kernel, iterations=1)
        
        # Ensure we don't accidentally grow the mask beyond the original prediction
        final_mask = cv2.bitwise_and(cleaned_mask, object_mask.astype(np.uint8))
        return final_mask

    def process_frame(self, hand_mask: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        """
        Main entry point to clean an object mask based on the hand mask.
        """
        clean_obj = object_mask.copy()

        if self.max_distance_px > 0:
            clean_obj = self.geofence_mask(hand_mask, clean_obj)

        if self.use_connected_components:
            clean_obj = self.filter_connected_components(hand_mask, clean_obj)

        return clean_obj
