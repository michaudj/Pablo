import multiprocessing

def square_worker(number):
    print(f"Processing {number}")
    return number * number

class NumberProcessor:
    def __init__(self, numbers):
        self.numbers = numbers

    def process_with_multiprocessing(self):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(square_worker, self.numbers)
        return results

if __name__ == '__main__':
    nums = list(range(10))
    processor = NumberProcessor(nums)
    output = processor.process_with_multiprocessing()
    print("Results:", output)
