import torch
import datetime
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

    
    def compare_models(model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')


    def get_datetime():
        return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")