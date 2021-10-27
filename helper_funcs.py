class HelperFuncs():

    def __init__(self):
        pass
        

    def print_dataset_lengths(datasets):
        for data in datasets:
            print(type(data))
            try:
                for i in range(100):
                    print(f"({i}) length: {len(data)} | type: {type(data)}")
                    data = data[0]
            except:
                print("------------------------------------------------------------------")