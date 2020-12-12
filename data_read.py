import os
import matplotlib.pyplot as plt


# 查看图片数量
def read_flower_data(folder_name):
    folders = os.listdir(folder_name)
    flower_names = []
    flower_nums = []
    for folder in folders:
        folder_path = os.path.join(folder_name, folder)
        images = os.listdir(folder_path)
        images_num = len(images)
        print("{}:{}".format(folder, images_num))
        flower_names.append(folder)
        flower_nums.append(images_num)

    return flower_names, flower_nums


# 绘制柱状图
def show_bar(x, y):
    # 绘图
    plt.barh(range(5), y, align='center', color='steelblue', alpha=0.8)
    # 添加轴标签
    plt.xlabel('num')
    # 添加标题
    plt.title('Num of flowers')
    # 添加刻度标签
    plt.yticks(range(5), x)
    # 设置Y轴的刻度范围
    # plt.xlim([32, 47])
    # 为每个条形图添加数值标签
    for x, y in enumerate(y):
        plt.text(y + 0.1, x, '%s' % y, va='center')
    # 显示图形
    plt.show()


if __name__ == '__main__':
    x, y = read_flower_data('../data/flower_photos')
    show_bar(x, y)