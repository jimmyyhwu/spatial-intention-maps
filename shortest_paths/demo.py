import time
import numpy as np
from skimage.draw import line
from PIL import Image
import shortest_paths

def visualize_path(configuration_space, path):
    image = 0.5 * configuration_space
    for i in range(len(path) - 1):
        rr, cc = line(*path[i], *path[i + 1])
        image[rr, cc] = 1
    Image.fromarray(np.round(255.0 * image).astype(np.uint8)).show()

def benchmark(fn_to_benchmark, num_runs=100):
    start_time = time.time()
    for _ in range(num_runs):
        fn_to_benchmark()
    avg_time = (time.time() - start_time) / num_runs
    print('Average time per run: {:.2f} ms ({})'.format(1000 * avg_time, fn_to_benchmark.__name__))

def benchmark_shortest_path(configuration_space, source, target):
    def shortest_path():
        graph = shortest_paths.GridGraph(configuration_space)
        graph.shortest_path(source, target)
    benchmark(shortest_path)

def benchmark_shortest_path_distance(configuration_space, source, target):
    def shortest_path_distance():
        graph = shortest_paths.GridGraph(configuration_space)
        graph.shortest_path_distance(source, target)
    benchmark(shortest_path_distance)

def benchmark_shortest_path_image(configuration_space, source):
    def shortest_path_image():
        graph = shortest_paths.GridGraph(configuration_space)
        graph.shortest_path_image(source)
    benchmark(shortest_path_image)

def main():
    # Construct graph
    configuration_space = np.load('sample-configuration-space.npy').astype(np.uint8)
    graph = shortest_paths.GridGraph(configuration_space)

    # Shortest path
    source, target = (75, 156), (131, 112)
    path = graph.shortest_path(source, target)
    correct_path = np.array([[ 75, 156], [98, 93], [110,  81], [118,  80], [124,  84], [131, 112]])
    assert np.allclose(np.array(path), correct_path, atol=2)
    print('Shortest path:', path)
    visualize_path(configuration_space, path)

    # Shortest path distance
    #print(np.sum([np.linalg.norm(correct_path[i + 1] - correct_path[i]) for i in range(len(correct_path) - 1)]))
    print('Shortest path distance:', graph.shortest_path_distance(source, target))

    # Shortest path distances to all pixels
    Image.fromarray(np.round(graph.shortest_path_image(source)).astype(np.uint8)).show()

    # Benchmarking
    print()
    benchmark_shortest_path(configuration_space, source, target)
    benchmark_shortest_path_distance(configuration_space, source, target)
    benchmark_shortest_path_image(configuration_space, source)

main()
