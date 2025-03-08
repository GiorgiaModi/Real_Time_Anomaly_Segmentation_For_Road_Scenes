import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import random

from ood_metrics import aupr, auroc, fpr_at_95_tpr
import cv2

from PIL import Image, ImageDraw, ImageFont
import os

from argparse import ArgumentParser

def generate_colormap():
    """ Generate a colormap that gradually goes from blue to white to red """

    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    # Gradually go from red (index=0) to white (index=128)
    for i in range(128):
        ratio = i / 127
        b = int(255 * ratio)
        g = int(255 * ratio)
        r = 255
        colormap[i, 0] = [b, g, r]
    # Then go from white (index=128) to blue (index=255)
    for i in range(128, 256):
        ratio = (i - 128) / 127
        b = 255
        g = int(255 * (1 - ratio))
        r = int(255 * (1 - ratio))
        colormap[i, 0] = [b, g, r]
    return colormap

def save_colored_score_image(image_path, anomaly_score, save_path, file_name):
    """
    Save the image with the anomaly score colored in a new image.
    
    image_path: path to the input image
    anomaly_score: anomaly score for each pixel
    save_path: path to save the colored image
    """
    
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (anomaly_score.shape[1], anomaly_score.shape[0]))
    
    # Normalize the anomaly score
    anomaly_score = (anomaly_score - np.min(anomaly_score)) / (np.max(anomaly_score) - np.min(anomaly_score))
    
    # Apply the colormap
    anomaly_score = cv2.applyColorMap((anomaly_score * 255).astype(np.uint8), generate_colormap())
    
    # Combine the original image and the colored anomaly score
    # combined = cv2.addWeighted(image, 0.5, anomaly_score, 0.5, 0)
    
    # Save the image
    cv2.imwrite(f"{save_path}/{file_name}.png", cv2.cvtColor(anomaly_score, cv2.COLOR_RGB2BGR))


def plot_roc(preds, labels, title="Receiver operating characteristic", save_path=None, file_name=None):
    """Plot an ROC curve based on unthresholded predictions and true binary labels.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    fpr, tpr, _ = roc_curve(labels, preds)

    # Compute FPR (95% TPR)
    tpr95 = fpr_at_95_tpr(preds, labels)

    # Compute AUROC
    roc_auc = auroc(preds, labels)

    # Draw the plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.2f' % tpr95)
    plt.plot([tpr95, tpr95], [0, 1], color='black', lw=lw, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}/{file_name}_roc_curve.png")
    else:
        plt.show()


def plot_pr(preds, labels, title="Precision recall curve", save_path=None, file_name=None):
    """Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PRC curve (area = %0.2f)' % prc_auc)
    #     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}/{file_name}_pr_curve.png")
    else:
        plt.show()


def plot_barcode(preds, labels, title="Barcode plot", save_path=None, file_name=None):
    """Plot a visualization showing inliers and outliers sorted by their prediction of novelty."""
    # the bar
    x = sorted([a for a in zip(preds, labels)], key=lambda x: x[0])
    x = np.array([[49, 163, 84] if a[1] == 1 else [173, 221, 142] for a in x])
    # x = np.array([a[1] for a in x]) # for bw image

    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary_r, interpolation='nearest')

    plt.title(title)
    fig = plt.figure()

    # a horizontal barcode
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
    ax.imshow(x.reshape((1, -1, 3)), **barprops)
    plt.tight_layout()
    
    
    if save_path is not None:
        plt.savefig(f"{save_path}/{file_name}_barcode_plot.png")
    else:
        plt.show()


def create_concatenated_image_with_titles(input_folder, output_image):
    font_size = 90  # Font size
    title_margin = 50  # Margin under the title

    images = []
    titles = []

    # Order the files in the folder
    sorted_files = sorted(os.listdir(input_folder), key=lambda x: x.lower())

    seen_files = set()  # Avoid duplicates

    # Ensure specific order for "image.png" and "ground_truth.png"
    priority_files = ["image.png", "ground_truth.png"]
    other_files = [f for f in sorted_files if f.lower() not in priority_files]
    random.shuffle(other_files)
    ordered_files = priority_files + other_files

    for file_name in ordered_files:
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            try:
                base_name, ext = os.path.splitext(file_name)
                ext = ext.lower()

                if ext == ".png" and base_name.lower() not in seen_files:
                    with Image.open(file_path) as img:
                        img = img.convert("RGBA")

                        if file_name.lower() == "image.png":
                            images.append(img)
                            titles.append("Image")
                            seen_files.add("image")
                        elif file_name.lower() == "ground_truth.png":
                            images.append(img)
                            titles.append("Ground Truth")
                            seen_files.add("ground_truth")
                        else:
                            images.append(img)
                            titles.append(base_name)  # Exact name of the file without extension
                            seen_files.add(base_name.lower())
            except Exception as e:
                print(f"Error loading file {file_name}: {e}")

    if not images:
        print("No images found in the folder. Exiting.")
        return

    # Calculate the dimensions of the final image
    width = sum(img.width for img in images)
    height = max(img.height for img in images) + font_size + title_margin + 20

    # Create a new blank image
    concatenated_image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(concatenated_image)

    # Load the font
    try:
        font_path = "/content/AnomalySegmentation/eval/Helvetica.ttc"  # Path to Helvetica font
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Font not found. Using default font.")
        font = ImageFont.load_default()

    # Position images and titles
    x_offset = 0
    for img, title in zip(images, titles):
        try:
            # Calculate text dimensions
            if font != ImageFont.load_default():
                bbox = draw.textbbox((0, 0), title, font=font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                text_width, text_height = draw.textsize(title, font=font)

            # Center the text with respect to the image
            text_x = x_offset + (img.width - text_width) // 2
            text_y = 10  # Top margin for the text
            draw.text((text_x, text_y), title, fill="black", font=font)

            # Add the image
            concatenated_image.paste(img, (x_offset, font_size + title_margin))
            x_offset += img.width
        except Exception as e:
            print(f"Error processing image {title}: {e}")

    # Save the concatenated image
    try:
        concatenated_image.save(output_image)
        print(f"Image saved as {output_image}")
    except Exception as e:
        print(f"Error saving image: {e}")



def main():
    parser = ArgumentParser()
    parser.add_argument('--name_dir',default="/content/AnomalySegmentation/visualization/baselines")
    parser.add_argument('--name_output', default="baseline_visualization.png")
    args = parser.parse_args()

    create_concatenated_image_with_titles(args.name_dir, args.name_output)

if __name__ == '__main__':
    main()
