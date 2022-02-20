#Daniel Haber
#322230020
import matplotlib.pyplot as plt
import numpy as np
import sys

MAX = 20


def average_loss(image, centroids, k):
    distance_sum = 0
    pixel_num = 0

    for pixel in image:
        min_distance = distance(centroids[0], pixel)
        for i in range(1, k):
            curr_distance = distance(centroids[i], pixel)
            if curr_distance < min_distance:
                min_distance = curr_distance
        pixel_num += 1
        min_distance = min_distance ** 2
        distance_sum += min_distance
    return distance_sum / pixel_num


def distance(centroid, pixel):
    dist = ((centroid[0] - pixel[0])**2 + (centroid[1] - pixel[1])**2 + (centroid[2] - pixel[2])**2)**0.5
    return dist


class Average:
    sum_r = 0
    sum_g = 0
    sum_b = 0
    n = 0


def k_means(image, centroids, k, out_fname):
    centroids_average = []
    XplotList = []
    YplotList = []
    iteration = -1
    for i in range(0, k):
        centroids_average.append(Average())
    while True:
        iteration += 1
        centroids_copy = centroids.copy()
        for pixel in image:
            min_distance = distance(list(centroids[0]), list(pixel))
            min_index = 0
            for i in range(1, k):
                curr_distance = distance(list(centroids[i]), list(pixel))
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    min_index = i
            centroids_average[min_index].sum_r += pixel[0]
            centroids_average[min_index].sum_g += pixel[1]
            centroids_average[min_index].sum_b += pixel[2]
            centroids_average[min_index].n += 1

        for i in range(0, k):
            if centroids_average[i].n != 0:
                centroids[i][0] = centroids_average[i].sum_r / centroids_average[i].n
                centroids[i][1] = centroids_average[i].sum_g / centroids_average[i].n
                centroids[i][2] = centroids_average[i].sum_b / centroids_average[i].n

        aveCost = average_loss(image, centroids, k)
        XplotList.append(iteration)
        YplotList.append(aveCost)

        for average in centroids_average:
            average.sum_r = 0
            average.sum_g = 0
            average.sum_b = 0
            average.n = 0
        centroids = centroids.round(4)
        out_fname.write(f"[iter {iteration}]:{','.join([str(i) for i in centroids])}\n")
        counter = 0
        for i in range(0, len(centroids)):
            if centroids[i][0] == centroids_copy[i][0] and centroids[i][1] == centroids_copy[i][1] and centroids[i][2] == centroids_copy[i][2]:
                counter += 1
        if counter == len(centroids) or iteration == MAX-1:
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title('Average loss as a function vs iterations, k=16')
            plt.plot(XplotList, YplotList)
            plt.show()
            break


def main():
    image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
    z = np.loadtxt(centroids_fname)  # load centroids
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255.
    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
    pixels = pixels.reshape(-1, 3)
    out = open(out_fname, "w")
    k_means(pixels, z, len(z), out)
    out.close()

if __name__ == "__main__":
    main()
