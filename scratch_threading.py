from concurrent.futures import ThreadPoolExecutor


def main():
    lst=[]
    executor = ThreadPoolExecutor(max_workers=10)
    def ff(x):
        lst.append(x)

    for i in range(1,100):
        #lst.append(i)
        executor.submit(ff, i)

    print(lst)


main()