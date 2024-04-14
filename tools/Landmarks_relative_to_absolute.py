def absolute_to_relative(Landmark_list, relative_point):
    """
    将三维相对坐标转化为绝对坐标
    Args:
        x_rel (float): 相对坐标的 x 分量
        y_rel (float): 相对坐标的 y 分量
        z_rel (float): 相对坐标的 z 分量
        x_ref (float): 参考点的 x 分量
        y_ref (float): 参考点的 y 分量
        z_ref (float): 参考点的 z 分量
    Returns:
        tuple: 绝对坐标的 x、y、z 分量
    """
    for landmark in Landmark_list:

        landmark[0] = landmark[0] - relative_point[0]
        landmark[1] = landmark[1] - relative_point[1]
        # landmark[2] = landmark[2] - relative_point[2]
    return Landmark_list