import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors


def common_draw(Ys: list, labels: list, y_tick_tuple: tuple, x_lim_tuple: tuple, x_name: list = None,
                saveName: str = "pic/paper/temp.pdf",
                colors=[],
                colorOffset=0,
                show_legend=True,
                legendsize=20,
                legend_pos=2,
                x_axis_name="",
                y_axis_name='Time (ms)',
                y_log=False,
                ymin=-1,
                ymax=-1,
                lengent_ncol=3,
                selfdef_figsize=(12, 6),
                BAR=True,
                bar_width=0.8,
                columnspacing=1.6,
                common_font_size=26,
                x_num=0,
                line=0,
                y_ticks=[],
                y_label_fontsize=''):
    with PdfPages(saveName) as pdf:
        font = {
            "weight": "normal",
            "size": common_font_size,
        }

        if x_num == 0:
            x_num = len(x_name)

        X = np.arange(x_num) + 1
        if colors == []:
            colors = ["#e64b35", "gold", "dodgerblue",
                      "lightgreen", "cyan", "green", "chocolate"]

        markers = ["o", "^", "D", "s", "*", "P", "x"]
        linestyles = ["-", ":", "-.", "--"]

        plt.figure(figsize=selfdef_figsize)
        if not BAR:
            for i in range(0, len(Ys)):
                plt.plot(X,
                         Ys[i],
                         label=labels[i],
                         linestyle=linestyles[i],
                         color=colors[i],
                         marker=markers[i],
                         markersize=10,
                         linewidth=3,
                         )
        else:
            X = np.arange(x_num) + 1
            total_width, n = bar_width, len(Ys)
            width = total_width / n
            X = X - (total_width - width) / 2
            for i in range(0, len(Ys)):
                plt.bar(X + width * i, Ys[i], width=width, label=labels[i],
                        color=colors[i % len(colors) + colorOffset],
                        edgecolor="black")

        xlim_left, xlim_right = x_lim_tuple
        if line != 0:
            line_x = np.linspace(xlim_left, xlim_right, 4)
            line_y = [line, line, line, line]
            plt.plot(line_x, line_y, color="darkgreen",
                     label="RTc3 Uniform",  linestyle='--')
        if y_log:
            plt.yscale("log")
        x_ticks = np.linspace(1, x_num, len(x_name))
        if x_name == None:
            x_name = [str(float(x)/10) for x in range(0, x_num)]
        plt.xticks(x_ticks, x_name, fontsize=common_font_size)

        # adaptively
        plt.yticks(fontsize=common_font_size)
        if y_ticks != []:
            plt.yticks(y_ticks[0], y_ticks[1])
        if ymax != -1:
            plt.ylim(ymax=ymax)
        if ymin != -1:
            plt.ylim(ymin=ymin)
        
        ax = plt.gca()
        if x_axis_name != "":
            ax.set_xlabel(x_axis_name, font)
        if y_label_fontsize == '':
            ax.set_ylabel(y_axis_name, font)
        else:
            ax.set_ylabel(
                y_axis_name, {"weight": "normal", "size": y_label_fontsize})
        if show_legend:
            ax.legend(prop={'size': legendsize}, loc=legend_pos,
                    ncol=lengent_ncol, columnspacing=columnspacing)
        plt.tight_layout()
        pdf.savefig()

def draw_line_chart(Ys: list, labels: list, x_name: list,
                    saveName="pic/paper/temp.pdf", colors=[],
                    legend_size=20, legend_pos=2,
                    x_axis_name="", y_axis_name="Time (ms)",
                    y_log=False, ymin=-1, ymax=-1,
                    lengent_ncol=3, selfdef_figsize=(12, 6),
                    columnspacing=1.6, common_font_size=26,
                    y_ticks=[], y_label_fontsize="", show_legend=True):
    with PdfPages(saveName) as pdf:
        font = {"weight": "normal", "size": common_font_size}
        x_num = len(x_name)
        X = np.arange(x_num) + 1
        markers = ["o", "^", "D", "s", "*", "P", "x"]
        linestyles = [(0, (3, 5, 1, 5, 1, 5)), ":", "-.", "--", "solid"]
        plt.figure(figsize=selfdef_figsize)
        for i in range(0, len(Ys)):
            plt.plot(X, Ys[i], label=labels[i], linestyle=linestyles[i], color=colors[i], marker=markers[i],
                     markersize=10, linewidth=3)
        if y_log:
            plt.yscale("log")
        x_ticks = np.linspace(1, x_num, len(x_name))
        if x_name == None:
            x_name = [str(float(x)/10) for x in range(0, x_num)]
        plt.xticks(x_ticks, x_name, fontsize=common_font_size)

        # adaptively
        plt.yticks(fontsize=common_font_size)
        if y_ticks != []:
            plt.yticks(y_ticks[0], y_ticks[1])
        if ymax != -1:
            plt.ylim(ymax=ymax)
        if ymin != -1:
            plt.ylim(ymin=ymin)

        ax = plt.gca()
        if x_axis_name != "":
            ax.set_xlabel(x_axis_name, font)
        if y_label_fontsize == "":
            ax.set_ylabel(y_axis_name, font)
        else:
            ax.set_ylabel(
                y_axis_name, {"weight": "normal", "size": y_label_fontsize})
        if show_legend:
            ax.legend(prop={'size': legend_size}, loc="upper left", bbox_to_anchor=(0, 1.25),
                      ncol=lengent_ncol, columnspacing=columnspacing)
        plt.tight_layout()
        pdf.savefig()

def draw_bar_hatch(Ys, labels, x_name,
                   save_name,
                   colors=[],
                   legend_size=20,
                   legend_pos=2,
                   x_axis_name="",
                   y_axis_name='Time (ms)',
                   y_log=False,
                   ymin=-1,
                   ymax=-1,
                   lengent_ncol=3,
                   selfdef_figsize=(12, 6),
                   bar_width=0.8,
                   columnspacing=1.6,
                   common_font_size=26,
                   x_num=0,
                   y_ticks=[],
                   y_label_fontsize=''):
    with PdfPages(save_name) as pdf:
        font = {
            "weight": "normal",
            "size": common_font_size,
        }
        if x_num == 0:
            x_num = len(x_name)
        hatches = ['//','--||','\\\\','--','o', 'x',  '+', '*', 'O', ]
        total_width, n = bar_width, len(Ys)
        width = total_width / n
        X = np.arange(x_num) + 1
        X = X - (total_width - width) / 2
        plt.figure(figsize=selfdef_figsize)
        for i in range(0, len(Ys)):
            plt.bar(X + width * i, Ys[i], width, label=labels[i],
                    color=colors[i],
                    edgecolor="black",
                    hatch=hatches[i])
        if y_log:
            plt.yscale("log")
        x_ticks = np.linspace(1, x_num, len(x_name))
        if x_name == None:
            x_name = [str(float(x)/10) for x in range(0, x_num)]
        plt.xticks(x_ticks, x_name, fontsize=common_font_size)
        plt.yticks(fontsize=common_font_size)
        if y_ticks != []:
            plt.yticks(y_ticks[0], y_ticks[1])
        if ymax != -1:
            plt.ylim(ymax=ymax)
        if ymin != -1:
            plt.ylim(ymin=ymin)
        ax = plt.gca()
        if x_axis_name != "":
            ax.set_xlabel(x_axis_name, font)
        if y_label_fontsize == "":
            ax.set_ylabel(y_axis_name, font)
        else:
            ax.set_ylabel(
                y_axis_name, {"weight": "normal", "size": y_label_fontsize})
        ax.legend(prop={'size': legend_size}, loc=legend_pos,
                  ncol=lengent_ncol, columnspacing=columnspacing)
        plt.tight_layout()
        pdf.savefig()

def draw_memory(Ys: list, labels: list, y_tick_tuple: tuple, x_lim_tuple: tuple, x_name: list = None,
                saveName: str = "pic/paper/temp.pdf",
                colors=[],
                colorOffset=0,
                legendsize=20,
                legend_pos=2,
                x_axis_name="",
                y_axis_name='Time (ms)',
                y_log=False,
                ymin=-1,
                ymax=-1,
                lengent_ncol=3,
                selfdef_figsize=(12, 6),
                BAR=True,
                bar_width=0.8,
                columnspacing=1.6,
                common_font_size=26,
                x_num=0,
                y_ticks=[],
                y_label_fontsize=''):
    with PdfPages(saveName) as pdf:
        font = {
            "weight": "normal",
            "size": common_font_size,
        }
        if x_num == 0:
            x_num = len(x_name)
        X = np.arange(x_num) + 1
        plt.figure(figsize=selfdef_figsize)
        total_width, n = bar_width, len(Ys)
        width = total_width / n
        X = X - (total_width - width) / 2
        handles = []
        for i in range(0, n - 1):
            h = plt.bar(X + width * i, Ys[i], width=width, label=labels[i],
                    color=colors[i % len(colors) + colorOffset],
                    edgecolor="black")
            handles.append(h)
        h = plt.bar(X + width * (n - 2), Ys[n - 1], bottom=Ys[n - 2], width=width, label=labels[n - 1],
                color=colors[n - 1],
                edgecolor="black")
        handles.append(h)
        if y_log:
            plt.yscale("log")
        x_ticks = np.linspace(1, x_num, len(x_name))
        if x_name == None:
            x_name = [str(float(x)/10) for x in range(0, x_num)]
        plt.xticks(x_ticks, x_name, fontsize=common_font_size)

        # adaptively
        plt.yticks(fontsize=common_font_size)
        if y_ticks != []:
            plt.yticks(y_ticks[0], y_ticks[1])
        if ymax != -1:
            plt.ylim(ymax=ymax)
        if ymin != -1:
            plt.ylim(ymin=ymin)
        ax = plt.gca()
        if x_axis_name != "":
            ax.set_xlabel(x_axis_name, font)
        if y_label_fontsize == '':
            ax.set_ylabel(y_axis_name, font)
        else:
            ax.set_ylabel(
                y_axis_name, {"weight": "normal", "size": y_label_fontsize})
        
        # new_handles = [handles[0], handles[3], handles[1], handles[4], handles[2]]
        ax.legend(handles=handles, prop={'size': legendsize}, loc=legend_pos,
                  ncol=lengent_ncol, columnspacing=columnspacing)

        plt.tight_layout()
        pdf.savefig()

def draw_stacking(Ys: list, labels: list, x_lim_tuple: tuple, x_name: list = None,
                  saveName: str = "pic/paper/temp.pdf",
                  colors=[],
                  colorOffset=0,
                  legendsize=26,
                  legend_pos=2,
                  x_axis_name="Selectivity",
                  y_axis_name='Time (ms)',
                  y_log=False,
                  ymax=-1,
                  lengent_ncol=3,
                  selfdef_figsize=(12, 6),
                  BAR=True,
                  columnspacing=1.6,
                  common_font_size=26):
    with PdfPages(saveName) as pdf:
        font = {
            "weight": "normal",   # "bold"
            "size": common_font_size,
        }
        x_num = len(x_name)

        print(f"[INFO] use x_num: {x_num}")

        X = np.arange(x_num) + 1
        if colors == []:
            # colors = ["#e64b35","gold","dodgerblue","lightgreen","cyan","green","chocolate"]
            colors = ["lightgreen", 'xkcd:jade',
                      'dodgerblue', 'gold', '#e64b35']
        plt.figure(figsize=selfdef_figsize)
        X = np.arange(x_num) + 1
        width = 0.65
        print(len(Ys), len(X))
        print(labels)

        accumulated_height = []
        accumulated_height.append(Ys[0])
        for i in range(1, len(Ys)):
            cur_height = []
            for j in range(x_num):
                cur_height.append(Ys[i][j] + accumulated_height[i - 1][j])
            accumulated_height.append(cur_height)
        accumulated_height.insert(0, [0] * x_num)

        # set bottom
        for i in range(0, len(Ys)):
            plt.bar(X, Ys[i], width=width, label=labels[i],
                    bottom=accumulated_height[i],
                    color=colors[i % len(colors) + colorOffset],
                    edgecolor="black")

        xlim_left, xlim_right = x_lim_tuple
        if y_log:
            plt.yscale("log")
        print(plt.xlim(xlim_left, xlim_right))
        print(plt.ylim())
        x_ticks = np.linspace(1, x_num, x_num)
        print(x_ticks)
        if x_name == None:
            x_name = [str(float(x)/10) for x in range(0, x_num)]
        plt.xticks(x_ticks, x_name, fontsize=common_font_size)

        # adaptively
        plt.yticks(fontsize=common_font_size)
        if ymax != -1:
            plt.ylim(ymax=ymax)

        ax = plt.gca()
        if x_axis_name != "":
            ax.set_xlabel(x_axis_name, font)
        ax.set_ylabel(y_axis_name, font)
        ax.legend(prop={'size': legendsize}, loc=legend_pos,
                  ncol=lengent_ncol, columnspacing=columnspacing)
        plt.tight_layout()
        pdf.savefig()

def draw_vary_ray_num():
    Y = [
        [2063293.594, 18871485.19, 102663080.3, 139814773.4, 141472064],
    ]
    Y_label = ["STK"]
    X_tick_name = ["1k", "10k", "100k", "1000k", "3000k"] 
    common_draw(Y, Y_label, None, (0.5, 0.5 + len(X_tick_name)),
                X_tick_name, "pic/vary_ray_num.pdf",
                colors=["dodgerblue"],
                colorOffset=0,
                x_axis_name='Number of rays',
                y_axis_name='RPS',
                y_log=True,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=3,
                # ymax=140,
                BAR=False,
                selfdef_figsize=(8,6))

def time_distribution(colors):
    record_x_name = ['STK', 'RBF', 'TAO', 'GeoLife']
    record_labels = [
        'Transferring Data',
        'Updating Units',
        'Building BVH tree',
        'Identifying Cores',
        'Clustering',
    ]
    time = [
        [0.0479239, 0.0146973, 0.0139046, 0.0967642],
        [0.152525, 0.0895182, 0.0864128, 0.168669],
        [0.343448, 0.2078332, 0.245984, 0.787935],
        [0.0398505, 0.0580811, 0.0563122, 0.113625],
        [0.271402, 0.106079, 0.255189, 1.14684],
    ]
    draw_stacking(time,
                  record_labels,
                  x_lim_tuple=(0.5, 0.5 + len(record_x_name)),
                  x_name=record_x_name,
                  saveName="pic/breakdown-running-time.pdf",
                  colorOffset=0,
                  x_axis_name='Dataset',
                  y_axis_name='Time (ms)',
                  legendsize=24.5,
                  common_font_size=26,
                  lengent_ncol=2,
                #   ymax=0.73,
                  legend_pos='upper left',
                    columnspacing=0.4,
                  colors=colors,
                  )

def overall_time(colors):
    record_x_name = ["STK", "RBF", "TAO", "GeoLife"]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [940, 48.3, 266, 5020],
        [1411, 33.2, 252, 3943],
        [7.762652, 0.881185, 8.010665, 98.083338],
        [1.713436, 1.051327, 1.392579, 4.627078],
        [0.849248, 0.449634, 0.627923, 2.3125]
    ]
    draw_bar_hatch(
                total_time, record_labels, record_x_name, 
                "pic/baseline-time.pdf",
                colors=colors,
                selfdef_figsize=(12, 6),
                x_axis_name='Dataset',
                y_axis_name='Time (ms)',
                y_log=True,
                legend_size=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=2,
                legend_pos='upper left',
                ymax=6e6)

def optimization_effects(colors):
    record_x_name = ["STK", "RBF", "TAO", "GeoLife"]
    record_labels = ["Naive", "ICG", "ICG+CMRS"]
    total_time = [
        [16.6926, 1.08912, 7.19273, 83.7277],
        [10.641, 0.896635, 3.46354, 47.7756],
        [0.849248, 0.449634, 0.627923, 2.3125],
    ]
    draw_bar_hatch(
                total_time, record_labels, record_x_name, 
                "pic/optimization-effects.pdf",
                colors=colors,
                selfdef_figsize=(12, 6),
                x_axis_name='Dataset',
                y_axis_name='Time (ms)',
                y_log=True,
                legend_size=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                legend_pos='best')

def overall_memory(colors):
    record_x_name = ["GAU", "STK", "TAO"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD-Host", "RTOD-Device"]
    mem = [
        [83, 73, 44],
        [39, 36.6, 14.6],
        [42.5, 39.4, 14.6],
        [3.01172, 3.01953, 0.6875],
        [26, 26, 2],
    ]
    draw_memory(mem, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/highlight-results-mem-86.pdf",
                colors=colors,
                colorOffset=0,
                selfdef_figsize=(8, 6),
                x_axis_name='Dataset',
                y_axis_name='Memory (MB)',
                y_log=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=2,
                legend_pos='best',
                ymax=150)

def vary_window_size(colors):
    # STK
    record_x_name = ["50k", "100k", "200k", "400k"]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [334.26, 1385.01, 6264.21, 27488.2],
        [364.3809, 1358.243, 5235.04, 18545.031],
        [2.69084, 5.96088, 14.0751, 37.353],
        [1.05264, 1.62565, 2.29535, 3.81047],
        [0.554731, 0.82438, 1.31843, 2.39203],
    ]
    draw_line_chart(
                total_time, record_labels, record_x_name, 
                "pic/varying-window-STK.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='|Window|',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                columnspacing=0.8,
                show_legend=False,
                legend_size=26,
                lengent_ncol=2,
                legend_pos='upper right',
                # ymax=3000
                )

    # RBF
    record_x_name = ["2.5k", "5k", "10k", "20k"]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [7.75, 13.907, 33.5666, 88.6],
        [6.269, 11.657, 20.683, 49.65],
        [0.566347, 0.681071, 0.817033, 1.001],
        [0.803943, 0.843329, 0.858283, 0.8719],
        [0.392743, 0.412964, 0.461149, 0.481494],
    ]
    draw_line_chart(
                total_time, record_labels, 
                record_x_name, "pic/varying-window-RBF.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='|Window|',
                y_axis_name='Time (ms)',
                y_log=True,
                legend_size=26,
                common_font_size=26,
                columnspacing=0.8,
                show_legend=False,
                lengent_ncol=4,
                legend_pos='best',
                # ymax=200
                )

    # TAO
    record_x_name = ["5k", "10k", "20k", "40k"]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [69.025, 267.356, 864.996, 3025.85],
        [77.174, 247.722, 732.583, 2240.3146],
        [4.40209, 8.031, 14.0151, 25.1903],
        [2.24161, 1.401, 1.99919, 3.1056],
        [0.466426, 0.621629, 0.893692, 1.16426],
    ]
    draw_line_chart(
                total_time, record_labels,
                record_x_name, "pic/varying-window-TAO.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='|Window|',
                y_axis_name='Time (ms)',
                y_log=True,
                legend_size=26,
                common_font_size=26,
                columnspacing=0.8,
                show_legend=False,
                lengent_ncol=1,
                # ymax=12
                )
    
    # GeoLife
    record_x_name = ["50k", "100k", "200k", "400k"]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [1246.9, 6139.44, 27455.91, 122421.99],
        [1160.2, 4211.04, 15921.885, 76964],
        [21.4767, 45.0083, 98.133, 255.342],
        [2.10685, 3.19589, 4.65865, 7.88127],
        [1.07431, 1.52713, 2.28313, 3.83796],
    ]
    draw_line_chart(
                total_time, record_labels,
                record_x_name, "pic/varying-window-GeoLife.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='|Window|',
                y_axis_name='Time (ms)',
                y_log=True,
                legend_size=26,
                common_font_size=26,
                columnspacing=0.8,
                show_legend=False,
                lengent_ncol=1,
                # ymax=12
                )

def vary_slide_size(colors):
    # STK
    record_x_name = ["1k", "5k", "10k", "25k"]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [372.9409, 1690.053, 2984.202, 8541.027],
        [515.7911, 1745.899, 2720.9148, 6465.757],
        [5.98044, 5.92332, 6.01017, 5.96732],
        [1.59395, 1.60001, 1.61424, 1.64389],
        [0.755292, 0.82444, 0.868426, 0.974887],
    ]
    draw_line_chart(
                total_time, record_labels, record_x_name, 
                "pic/varying-stride-STK.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='|Stride| / |Window|',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False,
                )

    # RBF
    record_x_name = ["0.1k", "0.5k", "1k", "2.5k"]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [11.32, 46.267, 84.167, 237.167],
        [8.19333, 21.98333, 67.7333, 117.41666],
        [0.817853, 0.8168, 0.9819, 0.948417],
        [0.872673, 1.16488, 1.1764, 1.01625],
        [0.434283, 0.442253, 0.447933, 0.454834],
    ]
    draw_line_chart(
                total_time, record_labels, 
                record_x_name, "pic/varying-stride-RBF.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='|Stride| / |Window|',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False,
                )

    # TAO
    record_x_name = ["0.1k", "0.5k", "1k", "2.5k"]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [59.4784, 269.5769, 545.28, 1443.37],
        [67.9731, 270.529, 524.908, 1171.2389],
        [8.03219, 8.0389, 7.97935, 8.11054],
        [1.39837, 1.4036, 1.398, 1.38784],
        [0.622362, 0.62325, 0.625237, 0.635083],
    ]
    draw_line_chart(
                total_time, record_labels,
                record_x_name, "pic/varying-stride-TAO.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='|Stride| / |Window|',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False,
                )
    
    # GeoLife
    record_x_name = ["2k", "10k", "20k", "50k"]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [4079.491, 21431.15, 49539.46, 137434.7789],
        [3823.926, 20248.875, 38021.5109, 91006.5294],
        [97.5971, 97.964, 97.9732, 98.4212],
        [4.63031, 4.63569, 4.64033, 4.65877],
        [2.18612, 2.28853, 2.29613, 2.51697],
    ]
    draw_line_chart(
                total_time, record_labels,
                record_x_name, "pic/varying-stride-GeoLife.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='|Stride| / |Window|',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False,
                )

def vary_eps(colors):
    # STK
    record_x_name = [0.055, 0.065, 0.075, 0.085, 0.095]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [1195.91, 1346.77, 1519.18, 1750.52, 1870.042],
        [1146.619, 1339.79, 1547.899, 1790.037, 2043.915],
        [5.20666, 5.64935, 6.10378, 6.63931, 6.97539],
        [1.5063, 1.58251, 1.68328, 1.74032, 1.71261],
        [0.804921, 0.815514, 0.833546, 0.860947, 0.851131],
    ]
    draw_line_chart(
                total_time, record_labels, record_x_name, 
                "pic/varying-eps-STK.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='Distance threshold ($Eps$)',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False,
                )

    # RBF
    record_x_name = [0.015, 0.025, 0.035, 0.045, 0.055]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [25.8167, 23.133, 37.6, 38.75, 51.75],
        [13.7, 21.983, 29.967, 27.633, 34.9],
        [0.509383, 0.645233, 0.80475, 1.02282, 1.2284],
        [0.904483, 0.874467, 0.85675, 0.86805, 0.887],
        [0.476082, 0.458895, 0.440983, 0.445768, 0.494466],
    ]
    draw_line_chart(
                total_time, record_labels, 
                record_x_name, "pic/varying-eps-RBF.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='Distance threshold ($Eps$)',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False,
                )

    # TAO
    record_x_name = [1.15, 1.25, 1.35, 1.45, 1.55]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [258.9, 271.512, 280.35, 292.005, 309.92],
        [239.34, 248.8876, 280.423, 303.41, 322.09],
        [7.04825, 7.71373, 8.23476, 8.92675, 9.55517],
        [1.41774, 1.38415, 1.32734, 1.34561, 1.44231],
        [0.655035, 0.641256, 0.650129, 0.627236, 0.645862],
    ]
    draw_line_chart(
                total_time, record_labels,
                record_x_name, "pic/varying-eps-TAO.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='Distance threshold ($Eps$)',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False,
                )
    
    # GeoLife
    record_x_name = [0.005, 0.015, 0.025, 0.035, 0.045]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [13160.45, 32277.4, 63390, 97859, 120400],
        [8485.87, 28915.27, 56084.81, 80090.34, 105149.2],
        [60.7002, 138.463, 232.131, 300.701, 353.146],
        [3.83563, 5.29879, 6.35041, 6.70076, 7.60078],
        [1.95702, 3.16758, 4.10279, 4.38947, 5.68359],
    ]
    draw_line_chart(
                total_time, record_labels,
                record_x_name, "pic/varying-eps-GeoLife.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='Distance threshold ($Eps$)',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False)

def vary_min_pts(colors):
    # STK
    record_x_name = [2, 4, 6, 8, 10, 12, 14]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [1690.053, 1016.83, 979.85, 987.3, 969.3, 983, 991.2],
        [1745.899, 1335.619, 1315.05, 1324.7, 1326.46, 1334.6, 1318.31],
        [5.96724, 7.7207, 7.72984, 7.78175, 7.73749, 7.78556, 7.7608],
        [1.60426, 1.72324, 1.72556, 1.73936, 1.73732, 1.73156, 1.75957],
        [0.827799, 0.812525, 0.824028, 0.841418, 0.846191, 0.850138, 0.868429],
    ]
    draw_line_chart(
                total_time, record_labels, record_x_name, 
                "pic/varying-minpts-STK.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='Density threshold ($MinPts$)',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False,
                )

    # RBF
    record_x_name = [2, 4, 6, 8, 10, 12, 14]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [27.55, 46.267, 32, 28.5, 26.12, 24.08, 23.28],
        [24.33, 21.98333, 23.08, 17.67, 20.183, 23.483, 17.3333],
        [0.6585, 0.816417, 0.821617, 0.833967, 0.839917, 0.831467, 0.843667],
        [0.6893, 0.856033, 0.889883, 0.9242, 0.991083, 1.03213, 1.08822],
        [0.432817, 0.440833, 0.471619, 0.501432, 0.541671, 0.581148, 0.620549],
    ]
    draw_line_chart(
                total_time, record_labels, 
                record_x_name, "pic/varying-minpts-RBF.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='Density threshold ($MinPts$)',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False,
                )

    # TAO
    record_x_name = [2, 4, 6, 8, 10, 12, 14]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [239.4, 266.36, 269.5769, 257.8, 258.9, 260.2, 257.46],
        [247.653, 254.248, 270.529, 253.4, 254.17, 249.3, 250.574],
        [7.17507, 7.96892, 7.91603, 8.00291, 7.98473, 8.00285, 8.044],
        [0.999503, 1.34341, 1.39976, 1.44053, 1.47366, 1.50226, 1.52249],
        [0.61176, 0.612657, 0.621792, 0.638873, 0.65647, 0.66728, 0.678002],
    ]
    draw_line_chart(
                total_time, record_labels,
                record_x_name, "pic/varying-minpts-TAO.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='Density threshold ($MinPts$)',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False,
                )
    
    # GeoLife
    record_x_name = [2, 4, 6, 8, 10, 12, 14]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [22860, 25100, 21431.15, 24700, 28000, 24300, 25000],
        [16140, 20677, 20248.875, 15500, 20100, 19700, 18000],
        [80.9037, 97.9878, 97.7101, 97.9957, 97.7859, 97.7097, 98.1707],
        [3.60414, 4.44956, 4.65029, 4.79877, 4.99536, 5.09299, 5.20652],
        [2.33793, 2.28715, 2.28557, 2.28067, 2.2964, 2.32065, 2.3361],
    ]
    draw_line_chart(
                total_time, record_labels,
                record_x_name, "pic/varying-minpts-GeoLife.pdf",
                colors=colors,
                selfdef_figsize=(8, 6),
                x_axis_name='Density threshold ($MinPts$)',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                show_legend=False)

def vary_legend(colors):
    # record_x_name = ["10", "30", "50", "70", "100"]
    # record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    # total_time = [
    #     [117.1270265, 133.5844736, 195.2347218, 592.9089938, 3497.38145],
    #     [3.25, 4.08, 5.11, 7.05, 9.03],
    #     [17.66, 56.55, 165.67, 325.47, 538.45],
    #     [0.301978, 0.407293, 0.547744, 0.614817, 0.79909],
    # ]
    # common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
    #             record_x_name, "pic/varying-params-legend.pdf",
    #             colors=colors,
    #             colorOffset=0,
    #             selfdef_figsize=(12, 6),
    #             x_axis_name='Neighbor Threshold',
    #             y_axis_name='Time (ms)',
    #             y_log=True,
    #             show_legend=True,
    #             legendsize=26,
    #             common_font_size=26,
    #             columnspacing=0.8,
    #             lengent_ncol=4,
    #             ymax=100000,
    #             BAR=False)
    
    record_x_name = [2, 4, 6, 8, 10, 12, 14]
    record_labels = ["DISC", "DenForest", "FDBSCAN", "FDBSCAN-DenseBox", "RTDBStream"]
    total_time = [
        [22860, 25100, 21431.15, 24700, 28000, 24300, 25000],
        [16140, 20677, 20248.875, 15500, 20100, 19700, 18000],
        [80.9037, 97.9878, 97.7101, 97.9957, 97.7859, 97.7097, 98.1707],
        [3.60414, 4.44956, 4.65029, 4.79877, 4.99536, 5.09299, 5.20652],
        [2.33793, 2.28715, 2.28557, 2.28067, 2.2964, 2.32065, 2.3361],
    ]
    draw_line_chart(
                total_time, record_labels,
                record_x_name, "pic/varying-params-legend.pdf",
                colors=colors,
                selfdef_figsize=(20, 6),
                x_axis_name='Density threshold ($MinPts$)',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                legend_size=26,
                columnspacing=0.8,
                lengent_ncol=5)

# draw_vary_ray_num()
colors = ["#e6b745", "#e64b35", "xkcd:jade", "dodgerblue", "gold"]
overall_time(colors)
# vary_window_size(colors)
# vary_slide_size(colors)
# vary_eps(colors)
# vary_min_pts(colors)
# vary_legend(colors)
optimization_effects(colors)
# time_distribution(colors)
# overall_memory(colors + ["gold"])
