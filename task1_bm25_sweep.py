from main import main
import numpy as np
import matplotlib.pyplot as plt
import csv

def bm25_sweep():
    k1_range = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.2, 3.0])
    b_range = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ndcg_b = []
    precision_b = []
    ndcg_k1 = []
    precision_k1 = []
    
    k1_fixed = 0.9 # Pyserini default
    b_fixed = 0.4 # Pyserini default

    def test_a():
        print(f"Starting Test A: b with k1 set to any fixed value")
        for b in b_range:
            evals = main(k1=k1_fixed, b=b)
            ndcg_b.append(float(evals[0]))
            precision_b.append(float(evals[1]))

    def test_b():
        print(f"Starting Test B: k1 with b set to any fixed value")
        for k1 in k1_range:
            evals = main(k1=k1, b=b_fixed)
            ndcg_k1.append(float(evals[0]))
            precision_k1.append(float(evals[1]))

    def plot_test_a(b_range=b_range, ndcg_b=ndcg_b, precision_b=precision_b, k1_fixed=k1_fixed):
        plt.figure()
        plt.plot(b_range, ndcg_b, marker='o', label='NDCG@10')
        plt.plot(b_range, precision_b, marker='s', label='Precision@10')
        plt.xlabel('b')
        plt.ylabel('Score')
        plt.title(f'NDCG@10 and Precision@10 vs. b (k1={k1_fixed})')
        plt.legend()
        plt.grid(True)
        plt.savefig('outputs/task1a_bm25_ndcg_precision_vs_b.png')

        print(f"Generating test A plots")

        plt.close()

    def plot_test_b(k1_range=k1_range, ndcg_k1=ndcg_k1, precision_k1=precision_k1, b_fixed=b_fixed):
        plt.figure()
        plt.plot(k1_range, ndcg_k1, marker='o', label='NDCG@10')
        plt.plot(k1_range, precision_k1, marker='s', label='Precision@10')
        plt.xlabel('k1')
        plt.ylabel('Score')
        plt.title(f'NDCG@10 and Precision@10 vs. k1 (b={b_fixed})')
        plt.legend()
        plt.grid(True)
        plt.savefig('outputs/task1b_bm25_ndcg_precision_vs_k1.png')

        print(f"Generating test B plots")

        plt.close()

    def save_to_csv(param_name, param_values, ndcg_scores, precision_scores, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([param_name, 'ndcg@10', 'precision@10'])
            for p, n, prec in zip(param_values, ndcg_scores, precision_scores):
                writer.writerow([p, n, prec])


    test_a()
    test_b()

    plot_test_a()
    plot_test_b()

    save_to_csv('b', b_range, ndcg_b, precision_b, 'outputs/task1a_bm25_b.csv')
    save_to_csv('k1', k1_range, ndcg_k1, precision_k1, 'outputs/task1b_bm25_k1.csv')

    print(f"Returned: {ndcg_b}, {precision_b}, {ndcg_k1}, {precision_k1}")

if __name__ == "__main__":
    bm25_sweep()