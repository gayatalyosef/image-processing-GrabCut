import os
import numpy
import numpy as np
import cv2
import igraph
import argparse

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

GRAPH_OBJ = None
LAST_ENERGY = -1


import time

def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(f'{func.__name__} took {(t2 - t1)}')
        return res
    return wrapper

# -------------------------------- GMM objects ----------------------------------------


class GMMComponent:
    def __init__(self, pixels, total_n_pixels):
        self.total_n_pixels = total_n_pixels
        self.pixels = pixels
        self.n_pixels = pixels.shape[0]

        self.mean = np.mean(pixels, axis=0)
        self.pi = self.n_pixels / self.total_n_pixels
        self.covariance_matrix = self.calculate_covariance_matrix()
        self.det = np.linalg.det(self.covariance_matrix)

    def calculate_covariance_matrix(self):
        if self.n_pixels == 0:
            mat = np.zeros((3, 3))
        else:
            mat = cv2.calcCovarMatrix(
                samples=self.pixels,
                mean=self.mean,
                flags=cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE
            )[0].T

        if np.linalg.det(mat) == 0:
            mat = mat + np.eye(3) * 0.0001

        return mat

    def calculate_pdfs(self, pixels):
        delta = pixels - self.mean
        return (self.pi / np.sqrt(self.det)) * numpy.exp(
            -0.5 * np.einsum(
                'ij,ij->i', delta, numpy.matmul(np.linalg.inv(self.covariance_matrix), delta.T).T)
        )


class GMM:
    def __init__(self, pixels, n_components=5):
        self.pixels = pixels
        self.total_n_pixels = pixels.shape[0]
        self.n_components = n_components
        self.components = []
        self.labels = self.kmeans(self.pixels).flatten()
        self.Dns = None

        for center_id in range(self.n_components):
            self.components.append(GMMComponent(self.pixels[self.labels == center_id], self.total_n_pixels))

    def kmeans(self, data, max_iter=100, epsilon=0.1):
        _, labels, _ = cv2.kmeans(
            data=data,
            K=self.n_components,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon),
            attempts=max_iter,
            flags=cv2.KMEANS_RANDOM_CENTERS,
            bestLabels=None
        )
        return labels

    def find_component_id(self):
        probs = np.zeros((self.total_n_pixels, self.n_components))
        for i, component in enumerate(self.components):
            probs[:, i] = component.calculate_pdfs(self.pixels).T
        return np.argmax(probs, axis=1)

    def calculate_Dns(self, pixels):
        if self.Dns:
            return self.Dns

        d = np.zeros((pixels.shape[0]))
        for i, component in enumerate(self.components):
            d += component.calculate_pdfs(pixels).T
        self.Dns = -1 * numpy.log(d)

    def update(self, pixels):
        self.pixels = pixels
        self.total_n_pixels = pixels.shape[0]
        self.labels = self.find_component_id()
        self.Dns = None
        self.components = []
        for center_id in range(self.n_components):
            component = GMMComponent(self.pixels[self.labels == center_id], self.total_n_pixels)
            self.components.append(component)

# ------------------------------------------------------------------------------------------------


class Graph:
    def __init__(self, img, mask, bgGMM, fgGMM):
        self.img = np.float32(img)
        self.flat_img = self.img.reshape((-1, 3))

        self.mask = mask.reshape(-1)
        self.bgGMM = bgGMM
        self.fgGMM = fgGMM
        self.v = img.shape[0] * img.shape[1] + 1
        self.v_size = self.v + 1
        self.source_v = self.v_size - 2
        self.target_v = self.v_size - 1

        self.neighbor_edges = self.init_neighbors_edges()
        self.e = self.init_edges()

        self.beta = self.calculate_beta()
        self.Nsums, self.neighbors_capacities = self.calculate_neighbors_capacities()
        self.K = self.calculate_K()

    def update_gmms(self, bgGMM, fgGMM):
        self.bgGMM = bgGMM
        self.fgGMM = fgGMM

    def get_8_neighbors(self, pixels, i, j):
        rows, cols = pixels.shape
        neighbors = []

        # Get top neighbor
        if i > 0:
            neighbors.append(pixels[i - 1, j])

        # Get bottom neighbor
        if i < rows - 1:
            neighbors.append(pixels[i + 1, j])

        # Get left neighbor
        if j > 0:
            neighbors.append(pixels[i, j - 1])

        # Get right neighbor
        if j < cols - 1:
            neighbors.append(pixels[i, j + 1])

        # Get top-left neighbor
        if i > 0 and j > 0:
            neighbors.append(pixels[i - 1, j - 1])

        # Get top-right neighbor
        if i > 0 and j < cols - 1:
            neighbors.append(pixels[i - 1, j + 1])

        # Get bottom-left neighbor
        if i < rows - 1 and j > 0:
            neighbors.append(pixels[i + 1, j - 1])

        # Get bottom-right neighbor
        if i < rows - 1 and j < cols - 1:
            neighbors.append(pixels[i + 1, j + 1])

        return neighbors

    def init_neighbors_edges(self):
        self.neighbor_edges = []

        pixels = np.arange(self.v_size - 2).reshape(self.img.shape[:-1])

        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                _edges = [(pixels[i][j], n) for n in self.get_8_neighbors(pixels, i, j)]
                self.neighbor_edges.extend(_edges)
        return self.neighbor_edges

    def init_edges(self):
        source_edges = [(self.source_v, i) for i in range(self.v_size-2)]
        target_edges = [(i, self.target_v) for i in range(self.v_size-2)]
        return source_edges + target_edges + self.neighbor_edges

    def calculate_beta(self):
        rows, cols = self.img.shape[0], self.img.shape[1]
        left_diff = img[:, 1:, :] - img[:, :-1, :]
        right_diff = -left_diff
        up_diff = img[1:, :, :] - img[:-1, :, :]
        down_diff = -up_diff
        upleft_diff = img[1:, 1:, :] - img[:-1, :-1, :]
        downright_diff = -upleft_diff
        upright_diff = img[1:, -1:, :] - img[:-1, :1, :]
        downleft_diff = -upright_diff

        # Compute the sum of the squared differences between neighboring pixels
        beta_numerator = np.sum(np.square(left_diff)) + np.sum(np.square(upleft_diff)) + \
                         np.sum(np.square(up_diff)) + np.sum(np.square(upright_diff)) + \
                         np.sum(np.square(down_diff)) + np.sum(np.square(downright_diff)) + \
                         np.sum(np.square(right_diff)) + np.sum(np.square(downleft_diff))
        beta_denominator = 8 * rows * cols - 6 * (rows + cols) + 4
        beta = beta_denominator / (2 * beta_numerator)

        return beta

    def calculate_N(self, n, m):
        def dist(n, m):
            columns = self.img.shape[1]
            return np.linalg.norm(np.array([(n-1) // columns, (n-1) % columns]) - np.array([(m-1) // columns, (m-1) % columns]))

        delta = (self.flat_img[n] - self.flat_img[m])
        return (50 / dist(n, m)) * np.exp(-1 * self.beta * np.square(np.linalg.norm(delta)))

    def calculate_neighbors_capacities(self):
        Nsums = np.zeros(self.flat_img.shape[0])
        neighbors_capacities = []
        for (m, n) in self.neighbor_edges:
            N = self.calculate_N(m, n)
            neighbors_capacities.append(N)
            Nsums[m] += N
        return Nsums, neighbors_capacities

    def calculate_t_link_capacities(self):
        source_capacities = []
        target_capacities = []

        self.bgGMM.calculate_Dns(self.flat_img)
        self.fgGMM.calculate_Dns(self.flat_img)

        for pixel_number in range(self.v_size-2):
            if self.mask[pixel_number] == GC_FGD:
                source_capacity = self.K
                target_capacity = 0
            elif self.mask[pixel_number] == GC_BGD:
                source_capacity = 0
                target_capacity = self.K
            else:
                source_capacity = self.bgGMM.Dns[pixel_number]
                target_capacity = self.fgGMM.Dns[pixel_number]

            source_capacities.append(source_capacity)
            target_capacities.append(target_capacity)

        return source_capacities, target_capacities

    def calculate_capacities(self):
        source_capacities, target_capacities = self.calculate_t_link_capacities()
        return source_capacities + target_capacities + self.neighbors_capacities

    def create_graph(self):
        graph = igraph.Graph(self.v_size)
        graph.add_edges(self.e)
        return graph

    @timing_val
    def min_cut(self):
        capacities = self.calculate_capacities()
        graph = self.create_graph()
        min_cut = graph.st_mincut(self.source_v, self.target_v, capacities)
        return min_cut.partition[0], min_cut.partition[1], min_cut.value

    def calculate_K(self):
        return max(self.Nsums)
@timing_val
# Define the GrabCut algorithm function
def grabcut(img, rect, n_componennts=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    print(x,y,w,h)
    w -= x
    h -= y
    print(w,h)


    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask, n_componennts)

    num_iters = 100
    for i in range(num_iters):

        print(f'Iteration: {i}')

        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    mask[mask == GC_PR_BGD] = GC_BGD  # We decided to consider GC_PR_BGD as BGD
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components=5):
    # one big nparray for every color - RGB
    img = np.float32(img)

    background_pixels = img[numpy.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    foreground_pixels = img[numpy.logical_or(mask == GC_FGD, mask == GC_PR_FGD)]

    bgGMM = GMM(background_pixels, n_components)
    fgGMM = GMM(foreground_pixels, n_components)

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    img = np.float32(img)

    background_pixels = img[numpy.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    bgGMM.update(background_pixels)

    foreground_pixels = img[numpy.logical_or(mask == GC_FGD, mask == GC_PR_FGD)]
    fgGMM.update(foreground_pixels)

    global GRAPH_OBJ
    if not GRAPH_OBJ:
        GRAPH_OBJ = Graph(img, mask, bgGMM, fgGMM)
    else:
        GRAPH_OBJ.update_gmms(bgGMM, fgGMM)


    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    global GRAPH_OBJ
    source, target, energy = GRAPH_OBJ.min_cut()
    min_cut = [source, target]
    return min_cut, energy


def update_mask(mincut_sets, mask):
    global GRAPH_OBJ
    rows, columns = mask.shape

    fg_v = mincut_sets[0]

    unknown_indexes = np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD))
    img_indexes = np.arange(rows * columns, dtype=np.uint32).reshape(rows, columns)
    mask[unknown_indexes] = np.where(np.isin(img_indexes[unknown_indexes], fg_v), GC_PR_FGD, GC_PR_BGD)

    return mask


def check_convergence(energy):
    global LAST_ENERGY
    print(f'Energy is: {energy}')

    if np.abs(LAST_ENERGY - energy) < 1000:
        res = True
    else:
        res = False

    LAST_ENERGY = energy
    return res

def cal_metric(predicted_mask, gt_mask):
    num_total_pixels = gt_mask.size
    num_correct_pixels = np.sum(gt_mask == predicted_mask)

    # Compute the accuracy
    accuracy = num_correct_pixels / num_total_pixels

    # Compute the intersection and union of the predicted and ground truth regions
    intersection = np.sum(np.logical_and(gt_mask, predicted_mask))
    union = np.sum(np.logical_or(gt_mask, predicted_mask))

    # Compute the Jaccard similarity
    jaccard_similarity = intersection / union

    return accuracy, jaccard_similarity

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    parser.add_argument('--n_componnents', type=int, default=5, help='number of componnents')
    parser.add_argument('--blur_level', type=int, default=0, help='blur level of the source img before C')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # add blur
    if args.blur_level != 0:
        img = cv2.blur(img, (args.blur_level, args.blur_level))


    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect, args.n_componnents)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])

    from PIL import Image
    # im = Image.fromarray(mask*255)
    # im.save(f'/Users/oferbudin/Downloads/hw1/hw1/data/masks/{args.input_name}.bmp')
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
