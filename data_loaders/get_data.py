from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate

def get_dataset_class(name):
    if name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name.startswith('brics-hands'):
        from data_loaders.hands_datasets.brics_hands_dataset import BricsHands
        return BricsHands
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    # if hml_mode == 'gt':
    #     from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
    #     return t2m_eval_collate
    # if name in ["humanml", "kit"]:
    #     return t2m_collate
    # else:
    #     return all_collate
    return t2m_collate

def get_dataset(args, name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name.startswith('brics-hands'):
        sp = True if 'SP' in name else False # use single person data
        ss = True if 'SS' in name else False # use single scene data
        anno = True if 'ANNO' in name else False
        dataset = DATA(args=args, split=split, num_frames=num_frames, mode=hml_mode, single_person=sp, single_scene=ss, use_annotated_text = anno)
        
    elif name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(args, name, batch_size, num_frames, split='train', hml_mode='train'):
    dataset = get_dataset(args, name, num_frames, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader