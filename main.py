from fire import Fire
from glamor.train.scripts import train_glamor_atari, train_glamor_control

if __name__ == '__main__':
    Fire({
        'train_glamor_atari': train_glamor_atari,
        'train_glamor_control': train_glamor_control,
    })
