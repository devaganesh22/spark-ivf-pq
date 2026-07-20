import h5py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import urllib.request
import os
import shutil

datasets = [
    {"name": "sift-128", "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5", "file": "sift-128-euclidean.hdf5"},
    {"name": "gist-960", "url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5", "file": "gist-960-euclidean.hdf5"},
    {"name": "glove-25", "url": "http://ann-benchmarks.com/glove-25-angular.hdf5", "file": "glove-25-angular.hdf5"},
    {"name": "glove-100", "url": "http://ann-benchmarks.com/glove-100-angular.hdf5", "file": "glove-100-angular.hdf5"},
    {"name": "fashion-mnist", "url": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5", "file": "fashion-mnist-784-euclidean.hdf5"},
    {"name": "mnist", "url": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5", "file": "mnist-784-euclidean.hdf5"},
    {"name": "deepimage", "url": "http://ann-benchmarks.com/deep-image-96-angular.hdf5", "file": "deep-image-96-angular.hdf5"}
]

base_dir = "ann-sample"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Download complete.")
    else:
        print(f"File {filename} already exists. Skipping download.")

def convert_to_parquet(hdf5_path, out_prefix):
    print(f"Opening {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        if 'train' in f:
            pq_path = f"{out_prefix}_base.parquet"
            if not os.path.exists(pq_path):
                print(f"Converting 'train' to {pq_path}...")
                train_data = f['train'][:]
                df_train = pd.DataFrame({'vector': list(train_data)})
                table_train = pa.Table.from_pandas(df_train)
                pq.write_table(table_train, pq_path)
            
        if 'test' in f:
            pq_path = f"{out_prefix}_query.parquet"
            if not os.path.exists(pq_path):
                print(f"Converting 'test' to {pq_path}...")
                test_data = f['test'][:]
                df_test = pd.DataFrame({'vector': list(test_data)})
                table_test = pa.Table.from_pandas(df_test)
                pq.write_table(table_test, pq_path)
            
        if 'neighbors' in f and 'distances' in f:
            pq_path = f"{out_prefix}_groundtruth.parquet"
            if not os.path.exists(pq_path):
                print(f"Converting ground truth to {pq_path}...")
                neighbors = f['neighbors'][:]
                distances = f['distances'][:]
                df_gt = pd.DataFrame({'neighbors': list(neighbors), 'distances': list(distances)})
                table_gt = pa.Table.from_pandas(df_gt)
                pq.write_table(table_gt, pq_path)

if __name__ == '__main__':
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    for ds in datasets:
        ds_dir = os.path.join(base_dir, ds["name"])
        if not os.path.exists(ds_dir):
            os.makedirs(ds_dir)
            
        hdf5_path = os.path.join(ds_dir, ds["file"])
        out_prefix = os.path.join(ds_dir, ds["name"])
        
        # Handle existing sift files from previous run
        old_hdf5 = ds["file"]
        if os.path.exists(old_hdf5) and not os.path.exists(hdf5_path):
            print(f"Moving {old_hdf5} to {hdf5_path}")
            shutil.move(old_hdf5, hdf5_path)
            
        old_prefix = "sift" if ds["name"] == "sift-128" else ds["name"]
        old_base = f"{old_prefix}_base.parquet"
        if os.path.exists(old_base) and not os.path.exists(f"{out_prefix}_base.parquet"):
            print(f"Moving parquet files for {ds['name']}...")
            shutil.move(old_base, f"{out_prefix}_base.parquet")
            shutil.move(f"{old_prefix}_query.parquet", f"{out_prefix}_query.parquet")
            shutil.move(f"{old_prefix}_groundtruth.parquet", f"{out_prefix}_groundtruth.parquet")

        download_file(ds["url"], hdf5_path)
        convert_to_parquet(hdf5_path, out_prefix)
    
    print("All datasets processed and organized!")
