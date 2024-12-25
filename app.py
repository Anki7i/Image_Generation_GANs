import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the generator model
@st.cache_resource
def load_generator(model_path):
    try:
        st.write(f"Loading model from {model_path}") 
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Generate images using the loaded generator
def generate_images(generator, latent_dim, num_images):
    noise = np.random.normal(0, 1, (num_images, latent_dim)) 
    generated_images = generator.predict(noise)
    return generated_images

# Streamlit app
def main():
    st.title("GAN Face Generator")
    st.sidebar.header("Settings")

    # Sidebar options
    model_path = st.sidebar.text_input("Generator Model Path", "generator.h5")
    latent_dim = st.sidebar.number_input("Latent Vector Dimension", value=100, step=1, min_value=1)
    num_images = st.sidebar.slider("Number of Images to Generate", 1, 10, 1)

    # Load model
    if st.sidebar.button("Load Generator"):
        st.session_state['generator'] = load_generator(model_path)
        if st.session_state['generator'] is not None:
            st.success("Generator model loaded successfully!")
        else:
            st.error("Failed to load the model.")

    # Generate images
    if st.button("Generate"):
        if 'generator' not in st.session_state or st.session_state['generator'] is None:
            st.error("Load the generator model first!")
        else:
            st.info("Generating images...")
            fake_images = generate_images(st.session_state['generator'], latent_dim, num_images)

            # Display images
            fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))  
            if num_images == 1:
                axs = [axs]
            for i, ax in enumerate(axs):
                ax.imshow((fake_images[i] + 1) / 2)  
                ax.axis('off')
            st.pyplot(fig)

if _name_ == "_main_":
    # Initialize session state
    if 'generator' not in st.session_state:
        st.session_state['generator'] = None
    main()