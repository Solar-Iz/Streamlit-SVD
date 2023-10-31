import streamlit as st
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Загрузка изображения по URL
st.title("Сжатие изображения с использованием SVD")

# Создаем состояние (state) для управления этапами
state = st.session_state
if 'loaded_image' not in state:
    state.loaded_image = None

image_url = st.text_input("Введите URL изображения")

if image_url:
    if st.button("Загрузить и отобразить изображение"):
        image = io.imread(image_url)
        
        is_color_image = len(image.shape) == 3
        
        if is_color_image:
            image = image[:, :, 0]
        else:
            image = io.imread(image_url, as_gray=True)

        # Отображение изображения
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray' if not is_color_image else None)
        ax.axis('off')
        st.pyplot(fig)
        
        # Сохраняем изображение и флаг цветного изображения в состоянии (state)
        state.loaded_image = image
        state.is_color_image = is_color_image

# Рассчитываем сингулярные значения
if state.loaded_image is not None:
    U, sing_values, V = np.linalg.svd(state.loaded_image)

    # Здесь добавим ползунок для выбора k
    top_k = st.slider("Выберите количество сингулярных чисел для сжатия", 1, len(sing_values), 1)

    if st.button("Сжать изображение"):
        # Создание сжатой версии изображения
        trunc_U = U[:, :top_k]
        trunc_sigma = np.diag(sing_values[:top_k])
        trunc_V = V[:top_k, :]
        trunc_image = trunc_U @ trunc_sigma @ trunc_V

        # Отображение сжатой версии изображения
        fig, ax = plt.subplots()
        ax.imshow(trunc_image, cmap='gray' if not state.is_color_image else None)
        ax.axis('off')
        st.pyplot(fig)

        # Рассчитываем долю k от всех сингулярных чисел
        x = round(top_k / len(sing_values) * 100, 2)
        st.write(f'Доля k составляет от всех сингулярных чисел: {x}%')