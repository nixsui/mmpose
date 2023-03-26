dataset_info = dict(
    dataset_name='coco_wholebody_partial',
    paper_info=dict(
        author='Jin, Sheng and Xu, Lumin and Xu, Jin and '
        'Wang, Can and Liu, Wentao and '
        'Qian, Chen and Ouyang, Wanli and Luo, Ping',
        title='Whole-Body Human Pose Estimation in the Wild',
        container='Proceedings of the European '
        'Conference on Computer Vision (ECCV)',
        year='2020',
        homepage='https://github.com/jin-s13/COCO-WholeBody/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_shoulder',
            id=1,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        2:
        dict(
            name='right_shoulder',
            id=2,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        3:
        dict(
            name='left_elbow',
            id=3,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        4:
        dict(
            name='right_elbow',
            id=4,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        5:
        dict(
            name='left_wrist',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        6:
        dict(
            name='right_wrist',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        7:
        dict(
            name='left_hip',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        8:
        dict(
            name='right_hip',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        9:
        dict(
            name='left_knee',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        10:
        dict(
            name='right_knee',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        11:
        dict(
            name='left_ankle',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        12:
        dict(
            name='right_ankle',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        13:
        dict(
            name='left_big_toe',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_big_toe'),
        14:
        dict(
            name='left_heel',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='right_heel'),
        15:
        dict(
            name='right_big_toe',
            id=15,
            color=[255, 128, 0],
            type='lower',
            swap='left_big_toe'),
        16:
        dict(
            name='right_heel',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_heel')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_ankle', 'left_big_toe'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('left_ankle', 'left_heel'), id=13, color=[0, 255, 0]),
        14:
        dict(
            link=('right_ankle', 'right_big_toe'), id=14, color=[255, 128, 0]),
        15:
        dict(link=('right_ankle', 'right_heel'), id=15, color=[255, 128, 0])
    },
    joint_weights=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0
    ],
    # 'https://github.com/jin-s13/COCO-WholeBody/blob/master/'
    # 'evaluation/myeval_wholebody.py#L175'
    sigmas=[
        0.026, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 
        0.087, 0.089, 0.089, 0.068, 0.066, 0.068, 0.066
    ])
