from dataset.base_dataset import BaseDataSet


def build_dataset(args, dataset_type, is_icl, demo_type, sample_rate, random_state):
    if is_icl:
        if demo_type == 'random':
            
            if dataset_type == "train":
                data = BaseDataSet(args, dataset_type="train", is_icl=True, demo_type='random', sample_rate=0.1,random_state=10)
                num_classes = data.get_num_classes()
                
            else:
                data = BaseDataSet(args, dataset_type="test", is_icl=True, demo_type='random', sample_rate=0.1,random_state=10)
                num_classes = data.get_num_classes()
                
        else:
            return False

    else:
        
        if dataset_type == "train":
            data = BaseDataSet(args, dataset_type="train", is_icl=False, demo_type='random', sample_rate=0.1,random_state=10)
            num_classes = data.get_num_classes()
            
        else:
            data = BaseDataSet(args, dataset_type="test", is_icl=False, demo_type='random', sample_rate=0.1,random_state=10)
            num_classes = data.get_num_classes()

    return data, num_classes