import os
import numpy as np
from PIL import Image
from neutral_network import Neural_Network
import streamlit as st
from streamlit_drawable_canvas import st_canvas

SIZE = 288

st.title('MNIST Digit Recognizer')
st.header('''
Try to write a digit!
''')

nn = Neural_Network(from_file=True)

mode = st.radio("Select operation:", ("Draw", "Transform"))
col1, col2 = st.columns(2)
with col1:
    st.write('Input Image:')
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=30,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw" if mode == 'Draw' else "transform",
        key='canvas')

with col2:
    if canvas_result.image_data is not None:
        img = Image.fromarray(
            canvas_result.image_data.astype('uint8')).convert('L').resize((28, 28))
        rescaled = img.resize((SIZE, SIZE), resample=Image.NEAREST)
        st.write('Model Input:')
        st.image(rescaled)

test_x = np.asarray(img, dtype=np.uint8).reshape(784)

# with col2:
if st.button('Predict'):
    val = nn.predict(test_x)
    st.success(f'Result: {np.argmax(val)}')
    st.bar_chart(val)
