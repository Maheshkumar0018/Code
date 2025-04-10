# # import cv2

# # def split_image_into_tiles(image):
# #     overlap = 0
# #     tile_size = 640
# #     tiles = []
# #     positions = []
# #     h,w = image.shape[:2]
# #     for y in range(0, h-overlap,tile_size-overlap):
# #         for x in range(0,w-overlap,tile_size-overlap):
# #             tile = image[y:y+ tile_size, x:x+tile_size]
# #             if tile.shape[0]==tile_size and tile.shape[1]==tile_size:
# #                 tiles.append(tile)
# #                 positions.append([x,y])
# #     return tiles, positions


# # base_img = cv2.imread('./pexels-francesco-ungaro-1525041.jpg')
# # print('base img shape: ', base_img.shape)

# # image = cv2.resize(base_img, (4000, 1000))  
# # print('resize img shape: ', image.shape)


# # tiles, positions = split_image_into_tiles(image)
# # print('positions:',positions)
# # print('tiles',len(tiles))


# # for tile in tiles:
# #     cv2.imshow("tile", tile)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()




# import cv2

# def split_image_into_exact_12_tiles(image, rows, cols, overlap_x, overlap_y):
#     h, w = image.shape[:2]
    
#     # Compute tile size with overlap
#     tile_w = (w + (cols - 1) * overlap_x) // cols
#     tile_h = (h + (rows - 1) * overlap_y) // rows

#     tiles = []
#     positions = []

#     for row in range(rows):
#         for col in range(cols):
#             x = col * (tile_w - overlap_x)
#             y = row * (tile_h - overlap_y)

#             # Ensure the tile fits inside the image
#             if x + tile_w > w:
#                 x = w - tile_w
#             if y + tile_h > h:
#                 y = h - tile_h

#             tile = image[y:y + tile_h, x:x + tile_w]
#             tiles.append(tile)
#             positions.append([x, y])

#     return tiles, positions

# # Load and resize image
# base_img = cv2.imread('./pexels-francesco-ungaro-1525041.jpg')
# print('Base image shape:', base_img.shape)

# image = cv2.resize(base_img, (4000, 1000))
# print('Resized image shape:', image.shape)

# # Configuration for 12 tiles (3 cols x 4 rows)
# rows = 4
# cols = 3
# overlap_x = 100  # horizontal overlap
# overlap_y = 50   # vertical overlap

# tiles, positions = split_image_into_exact_12_tiles(image, rows, cols, overlap_x, overlap_y)
# print('Tile positions:', positions)
# print('Number of tiles:', len(tiles))

# # Optional: Save or visualize tiles
# # for i, tile in enumerate(tiles):
# #     cv2.imwrite(f'tile_{i}.jpg', tile)



import cv2
import numpy as np

def split_image_into_exact_12_tiles(image, rows, cols, overlap_x, overlap_y):
    h, w = image.shape[:2]
    tile_w = (w + (cols - 1) * overlap_x) // cols
    tile_h = (h + (rows - 1) * overlap_y) // rows

    tiles = []
    positions = []

    for row in range(rows):
        for col in range(cols):
            x = col * (tile_w - overlap_x)
            y = row * (tile_h - overlap_y)

            # Ensure within bounds
            x = min(x, w - tile_w)
            y = min(y, h - tile_h)

            tile = image[y:y + tile_h, x:x + tile_w]
            tiles.append(tile)
            positions.append((x, y))

    return tiles, positions, (tile_w, tile_h)

def merge_tiles(tiles, positions, full_shape):
    merged = np.zeros(full_shape, dtype=np.uint8)
    count = np.zeros(full_shape[:2], dtype=np.float32)

    tile_h, tile_w = tiles[0].shape[:2]

    for tile, (x, y) in zip(tiles, positions):
        merged[y:y+tile_h, x:x+tile_w] += tile
        count[y:y+tile_h, x:x+tile_w] += 1

    # Avoid division by zero
    count = np.clip(count, 1, None)
    merged = (merged / count[..., None]).astype(np.uint8)
    return merged

# --- MAIN ---
base_img = cv2.imread('./pexels-francesco-ungaro-1525041.jpg')
image = cv2.resize(base_img, (4000, 1000))

rows, cols = 4, 3
overlap_x, overlap_y = 100, 50

tiles, positions, tile_size = split_image_into_exact_12_tiles(image, rows, cols, overlap_x, overlap_y)
reconstructed = merge_tiles(tiles, positions, image.shape)

print("Reconstructed image shape:", reconstructed.shape)
print("Original image shape:", base_img.shape)

# Stack original and merged side-by-side
# stacked = np.hstack((image, reconstructed))
# cv2.imshow("Original (Left) vs Reconstructed (Right)", stacked)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def convert_tile_bboxes_to_original(bboxes_list, positions):
    """
    Converts list of tile-wise bboxes to original image coordinates.
    
    Args:
        bboxes_list: List of lists of bounding boxes for each tile.
                     Each bbox is in (x1, y1, x2, y2) format.
        positions: List of (x_offset, y_offset) for each tile.
    
    Returns:
        List of bounding boxes in original image coordinates.
    """
    original_bboxes = []
    for bboxes, (x_offset, y_offset) in zip(bboxes_list, positions):
        for (x1, y1, x2, y2) in bboxes:
            orig_x1 = x1 + x_offset
            orig_y1 = y1 + y_offset
            orig_x2 = x2 + x_offset
            orig_y2 = y2 + y_offset
            original_bboxes.append((orig_x1, orig_y1, orig_x2, orig_y2))
    return original_bboxes


# Let's say tile 1 has 2 objects, tile 2 has 1, and so on...
tile_bboxes = [
    [(100, 50, 200, 150), (250, 100, 300, 180)],  # tile 0
    [(50, 20, 150, 120)],                         # tile 1
    [],                                           # tile 2
    [(10, 10, 50, 60)],                           # tile 3
    # ... for all 12 tiles
]

# `positions` is the output from `split_image_into_exact_12_tiles()`
original_bboxes = convert_tile_bboxes_to_original(tile_bboxes, positions)

for i, bbox in enumerate(original_bboxes):
    print(f"BBox {i}: {bbox}")
