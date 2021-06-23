import streamlit as st
from io import BytesIO
import tensorflow as tf
from PIL import Image


IMAGE_SHAPE = (224, 224)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Dogs and Cats Predictor")

def main():
    file = st.file_uploader("Upload file", type=["png", "jpeg", "jpg"])
    show_file = st.empty()

    # if not file:
        # show_file.info("Upload a picture of an animal you want to be predicted.")
        # return

    # content = file.getvalue()

    @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model("DOGvsCAT_Model")
        return model
    
    with st.spinner('Loading model into memmory...'):
        model = load_model()

    if not file:
        show_file.info("Upload a picture of an animal you want to be predicted.")
        return

    content = file.getvalue()

    classes = ['CAT', 'DOG']
    
    def load_and_prep_image(image):
         """
         Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape,, color_channels)
         """
         # Read in the image
         # img = tf.io.read_file(filename)
         # Decode the read file into a tensor
         image = tf.image.decode_image(image)
         # Resize the image  
         image = tf.image.resize(image, size=IMAGE_SHAPE)
         #Grayscale
         if image.shape[2] == 1:
             img = tf.image.grayscale_to_rgb(image)
         # Rescale the image (getting all values between 0 & 1)
          # image = image/255

         return image

    st.write("Predicted Class :")
    with st.spinner("Classifying....."):
         img = load_and_prep_image(content)
         label = model.predict(tf.expand_dims(img, axis=0))
         st.write(classes[int(tf.argmax(tf.squeeze(label).numpy()))])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption="Classifying the animal", use_column_width=True)

main()