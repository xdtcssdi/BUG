from tqdm import tqdm
if __name__ == '__main__':
    with tqdm(total=10000) as pbar:
        for i in range(10000):
            pbar.update(i)