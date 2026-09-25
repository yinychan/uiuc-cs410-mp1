from main import main
import csv
import os

# apnews
# k1=1.8, b=0.8
# set_qld(mu) Query Likelihood with Dirichlet smoothing (QLD), typical default value for mu is 1000 or 2500 depending on collection length
# set_rm3(fb_terms=10, fb_docs=10, original_query_weight=0.5, filter_terms), use_python

def compare_algos(cname="apnews"):
    k1, b = 1.8, 0.8
    mu = 1000

    results = {}

    # BM25 baseline
    print(f"Running BM25 with k1={k1}, b={b} on {cname}")

    ndcg, precision = main(cname=cname, k1=k1, b=b, algorithm="bm25", expansion=False, precision_cutoff=0)
    results["BM25"] = (ndcg, precision)

    # Adding RM3 psuedo-relevance feedback to BM25
    print(f"Running BM25 with RM3 on {cname}")
    ndcg, precision = main(cname=cname, k1=k1, b=b, algorithm="bm25", expansion=True, precision_cutoff=0)
    results["BM25 + RM3"] = (ndcg, precision)

    # QLD
    print(f"Running QLD with mu={mu} on {cname}")
    ndcg, precision = main(cname=cname, mu=mu, algorithm="qld", expansion=False, precision_cutoff=0)
    results["QLD"] = (ndcg, precision)

    # Summary
    print("\nSummary of Results:")
    for algo, (ndcg, precision) in results.items():
        print(f"{algo}: NDCG@10 = {ndcg:.4f}, Precision@10 = {precision:.4f}")

    # Save to CSV
    output_file = f"outputs/task2_compare_algos_{cname}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "NDCG@10", "Precision@10"])
        for algo, (ndcg, precision) in results.items():
            writer.writerow([algo, ndcg, precision])

if __name__ == "__main__":
    compare_algos()