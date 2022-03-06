from matplotlib import pyplot as plt
import numpy as np

plt.figure()
a = [1,2,5,6,9,11,15,17,18, 20]
labels = ['MBA_simple_0.95_200', 'MBA_EWMA_0.95_200_0.95',
       'HS_simple_0.95_200', 'HS_weighted_0.95_200_0.995',
       'MBA_map_fx_0.95_200', 'MBA_simple_0.99_200',
       'MBA_EWMA_0.99_200_0.95', 'HS_simple_0.99_200',
       'HS_weighted_0.99_200_0.995', 'MBA_map_fx_0.99_200']
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00',
                  '#000000']
# plt.hlines(1,1,20)  # Draw a horizontal line
plt.xticks(range(len(a)), labels, rotation = 25)
plt.bar(range(len(a)),a)

plt.show()