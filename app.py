import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import tempfile
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict

# Streamlit Page Configuration
st.set_page_config(page_title="Moon Detection App", layout="wide")
st.title("🌕 Moon Crater, Boulder & Landslide Detection - Multi-Image Analysis")
st.markdown("""
This app analyzes multiple lunar surface images using a YOLOv8 model to detect and compare:
- **Craters** (cyan boxes)
- **Boulders** (green boxes)
- **Sliding zones** (red dots, based on proximity/overlap rules)
""")

# Upload multiple images
uploaded_files = st.file_uploader("📤 Upload lunar images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Load model with caching
@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

# Load training results with caching
@st.cache_data
def load_training_results():
    try:
        return pd.read_csv("results.csv")
    except FileNotFoundError:
        return None

def process_single_image(image_path, model):
    """Process a single image and return detection results"""
    results = model(image_path, conf=0.1)
    result = results[0]
    
    # Load original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_orig = img_rgb.copy()
    
    # Detection data
    boulders, craters, confidences = [], [], []
    
    for box in result.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls_id = box
        cls_id = int(cls_id)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        radius = int((x2 - x1) / 2)
        confidences.append(conf)
        
        if result.names[cls_id] == "boulder":
            boulders.append((cx, cy))
            color = (0, 255, 0)  # Green
        elif result.names[cls_id] == "crater":
            craters.append((cx, cy, radius))
            color = (255, 255, 0)  # Cyan
        
        cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # Sliding zone logic
    landslide_zones = []
    
    for i in range(len(boulders)):
        for j in range(i + 1, len(boulders)):
            if np.linalg.norm(np.array(boulders[i]) - np.array(boulders[j])) < 30:
                mid = tuple(np.mean([boulders[i], boulders[j]], axis=0).astype(int))
                landslide_zones.append(mid)
    
    for bx, by in boulders:
        for cx, cy, r in craters:
            dist = np.linalg.norm(np.array([cx, cy]) - np.array([bx, by]))
            if r - 15 < dist < r + 15:
                landslide_zones.append((bx, by))
    
    for i in range(len(craters)):
        for j in range(i + 1, len(craters)):
            (x1, y1, r1), (x2, y2, r2) = craters[i], craters[j]
            dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
            if dist < (r1 + r2) * 0.6:
                mid = tuple(np.mean([[x1, y1], [x2, y2]], axis=0).astype(int))
                landslide_zones.append(mid)
    
    # Draw red dots for landslide zones
    for pt in landslide_zones:
        cv2.circle(img_rgb, pt, 5, (0, 0, 255), -1)
    
    return {
        'original_image': img_orig,
        'processed_image': img_rgb,
        'boulders': boulders,
        'craters': craters,
        'landslide_zones': landslide_zones,
        'confidences': confidences,
        'image_size': img_orig.shape
    }

model = load_model()
training_data = load_training_results()

if uploaded_files:
    # Process all uploaded images
    st.markdown("---")
    st.markdown("## 🔄 Processing Images...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = {}
    comparison_data = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f'Processing {uploaded_file.name}...')
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name
        
        # Process image
        result = process_single_image(image_path, model)
        all_results[uploaded_file.name] = result
        
        # Collect comparison data
        comparison_data.append({
            'Image': uploaded_file.name,
            'Craters': len(result['craters']),
            'Boulders': len(result['boulders']),
            'Landslide Zones': len(result['landslide_zones']),
            'Total Detections': len(result['confidences']),
            'Avg Confidence': np.mean(result['confidences']) if result['confidences'] else 0,
            'Min Confidence': min(result['confidences']) if result['confidences'] else 0,
            'Max Confidence': max(result['confidences']) if result['confidences'] else 0,
            'Image Width': result['image_size'][1],
            'Image Height': result['image_size'][0]
        })
        
        # Clean up temporary file
        os.remove(image_path)
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text('Processing complete!')
    progress_bar.empty()
    status_text.empty()
    
    # ========== SECTION 1: PROCESSED IMAGES OUTPUT (FIRST) ==========
    st.markdown("---")
    st.markdown("## 🖼️ Detection Results - Processed Images")
    
    # Quick summary metrics at the top
    comparison_df = pd.DataFrame(comparison_data)
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Images Processed", len(uploaded_files))
    with summary_col2:
        st.metric("Total Craters", comparison_df['Craters'].sum())
    with summary_col3:
        st.metric("Total Boulders", comparison_df['Boulders'].sum())
    with summary_col4:
        st.metric("Total Landslide Zones", comparison_df['Landslide Zones'].sum())
    
    # Display all processed images in a grid
    st.markdown("### 🎯 All Detection Results")
    
    # Create columns for images (2 per row)
    num_cols = 2
    num_rows = (len(all_results) + num_cols - 1) // num_cols
    
    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            img_idx = row * num_cols + col_idx
            if img_idx < len(all_results):
                img_name = list(all_results.keys())[img_idx]
                result = all_results[img_name]
                
                with cols[col_idx]:
                    st.markdown(f"**{img_name}**")
                    
                    # Show processed image
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(result['processed_image'])
                    ax.axis('off')
                    ax.set_title(f"Detections: {img_name}", fontsize=12)
                    
                    # Add legend
                    handles = [
                        Patch(facecolor='cyan', edgecolor='black', label='Crater'),
                        Patch(facecolor='green', edgecolor='black', label='Boulder'),
                        Patch(facecolor='red', edgecolor='black', label='Landslide Zone')
                    ]
                    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                              ncol=3, frameon=False, fontsize=10)
                    st.pyplot(fig)
                    
                    # Quick stats for this image
                    st.markdown(f"**Craters:** {len(result['craters'])} | **Boulders:** {len(result['boulders'])} | **Landslide Zones:** {len(result['landslide_zones'])}")
                    if result['confidences']:
                        st.markdown(f"**Avg Confidence:** {np.mean(result['confidences']):.3f}")
                    
                    st.markdown("---")
    
    # ========== SECTION 2: DETAILED ANALYSIS (SECOND) ==========
    st.markdown("---")
    st.markdown("## 📊 Detailed Analysis & Comparisons")
    
    # Expandable section for detailed analysis
    with st.expander("📈 View Detailed Statistics & Analysis", expanded=False):
        
        # Detailed comparison table
        st.markdown("### 📋 Detailed Comparison Table")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Extended summary statistics
        st.markdown("### 📈 Extended Summary Statistics")
        extended_col1, extended_col2, extended_col3, extended_col4 = st.columns(4)
        
        with extended_col1:
            st.metric("Avg Craters per Image", f"{comparison_df['Craters'].mean():.1f}")
            st.metric("Max Craters in Single Image", comparison_df['Craters'].max())
        
        with extended_col2:
            st.metric("Avg Boulders per Image", f"{comparison_df['Boulders'].mean():.1f}")
            st.metric("Max Boulders in Single Image", comparison_df['Boulders'].max())
        
        with extended_col3:
            st.metric("Avg Landslide Zones per Image", f"{comparison_df['Landslide Zones'].mean():.1f}")
            st.metric("Max Landslide Zones in Single Image", comparison_df['Landslide Zones'].max())
        
        with extended_col4:
            st.metric("Overall Avg Confidence", f"{comparison_df['Avg Confidence'].mean():.3f}")
            st.metric("Best Performing Image", comparison_df.loc[comparison_df['Total Detections'].idxmax(), 'Image'])
        
        # Visualization comparisons
        st.markdown("### 📊 Performance Comparison Charts")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Detection counts comparison
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            x = np.arange(len(comparison_df))
            width = 0.25
            
            ax1.bar(x - width, comparison_df['Craters'], width, label='Craters', color='cyan', alpha=0.7)
            ax1.bar(x, comparison_df['Boulders'], width, label='Boulders', color='green', alpha=0.7)
            ax1.bar(x + width, comparison_df['Landslide Zones'], width, label='Landslide Zones', color='red', alpha=0.7)
            
            ax1.set_xlabel('Images')
            ax1.set_ylabel('Detection Count')
            ax1.set_title('Detection Counts by Image')
            ax1.set_xticks(x)
            ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in comparison_df['Image']], rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with chart_col2:
            # Confidence distribution
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            ax2.bar(comparison_df['Image'], comparison_df['Avg Confidence'], alpha=0.7, color='purple')
            ax2.set_xlabel('Images')
            ax2.set_ylabel('Average Confidence')
            ax2.set_title('Average Confidence by Image')
            ax2.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in comparison_df['Image']], rotation=45)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Correlation analysis
        st.markdown("### 🔍 Correlation Analysis")
        correlation_col1, correlation_col2 = st.columns(2)
        
        with correlation_col1:
            # Correlation heatmap
            numeric_cols = ['Craters', 'Boulders', 'Landslide Zones', 'Total Detections', 'Avg Confidence']
            corr_matrix = comparison_df[numeric_cols].corr()
            
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title('Feature Correlation Matrix')
            st.pyplot(fig3)
        
        with correlation_col2:
            # Scatter plot: Craters vs Boulders
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            scatter = ax4.scatter(comparison_df['Craters'], comparison_df['Boulders'], 
                                c=comparison_df['Avg Confidence'], cmap='viridis', alpha=0.7, s=100)
            ax4.set_xlabel('Number of Craters')
            ax4.set_ylabel('Number of Boulders')
            ax4.set_title('Craters vs Boulders (colored by Avg Confidence)')
            plt.colorbar(scatter, ax=ax4, label='Avg Confidence')
            ax4.grid(True, alpha=0.3)
            
            # Add image names as annotations
            for i, txt in enumerate(comparison_df['Image']):
                ax4.annotate(txt[:8], (comparison_df['Craters'].iloc[i], comparison_df['Boulders'].iloc[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            st.pyplot(fig4)
    
    # ========== SECTION 3: INDIVIDUAL IMAGE COMPARISON (THIRD) ==========
    st.markdown("---")
    st.markdown("## 🔍 Individual Image Comparison")
    
    # Expandable section for individual comparisons
    with st.expander("🖼️ View Original vs Processed Comparisons", expanded=False):
        
        # Image selection
        selected_images = st.multiselect(
            "Select images to compare (Original vs Processed):",
            options=list(all_results.keys()),
            default=list(all_results.keys())[:2] if len(all_results) >= 2 else list(all_results.keys())
        )
        
        if selected_images:
            # Display selected images
            for img_name in selected_images:
                result = all_results[img_name]
                
                st.markdown(f"### 📸 {img_name}")
                
                # Image display
                img_col1, img_col2 = st.columns(2)
                
                with img_col1:
                    st.markdown("**Original Image**")
                    fig_orig, ax_orig = plt.subplots(figsize=(6, 6))
                    ax_orig.imshow(result['original_image'])
                    ax_orig.axis('off')
                    ax_orig.set_title(f"Original: {img_name}", fontsize=12)
                    st.pyplot(fig_orig)
                
                with img_col2:
                    st.markdown("**Detection Results**")
                    fig_proc, ax_proc = plt.subplots(figsize=(6, 6))
                    ax_proc.imshow(result['processed_image'])
                    ax_proc.axis('off')
                    ax_proc.set_title(f"Detections: {img_name}", fontsize=12)
                    
                    # Add legend
                    handles = [
                        Patch(facecolor='cyan', edgecolor='black', label='Crater'),
                        Patch(facecolor='green', edgecolor='black', label='Boulder'),
                        Patch(facecolor='red', edgecolor='black', label='Landslide Zone')
                    ]
                    ax_proc.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                                  ncol=3, frameon=False, fontsize=8)
                    st.pyplot(fig_proc)
                
                # Individual stats
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("Craters", len(result['craters']))
                with stats_col2:
                    st.metric("Boulders", len(result['boulders']))
                with stats_col3:
                    st.metric("Landslide Zones", len(result['landslide_zones']))
                with stats_col4:
                    if result['confidences']:
                        st.metric("Avg Confidence", f"{np.mean(result['confidences']):.3f}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                
                st.markdown("---")
    
    # ========== SECTION 4: TRAINING PERFORMANCE (FOURTH) ==========
    # Training results section (if available)
    if training_data is not None:
        st.markdown("---")
        st.markdown("## 🏋️ Model Training Performance")
        
        # Expandable section for training data
        with st.expander("📊 View Model Training Details", expanded=False):
            
            train_col1, train_col2 = st.columns(2)
            
            with train_col1:
                st.markdown("### 📈 Final Model Performance")
                final_epoch = training_data.iloc[-1]
                
                st.metric("Final Precision", f"{final_epoch['metrics/precision(B)']:.4f}")
                st.metric("Final Recall", f"{final_epoch['metrics/recall(B)']:.4f}")
                st.metric("Final mAP@0.5", f"{final_epoch['metrics/mAP50(B)']:.4f}")
                st.metric("Final mAP@0.5:0.95", f"{final_epoch['metrics/mAP50-95(B)']:.4f}")
                
                st.markdown("### 📋 Training Summary")
                st.markdown(f"**Total Epochs:** {len(training_data)}")
                st.markdown(f"**Total Training Time:** {final_epoch['time']:.1f} seconds")
                st.markdown(f"**Best mAP@0.5:** {training_data['metrics/mAP50(B)'].max():.4f}")
                st.markdown(f"**Final Learning Rate:** {final_epoch['lr/pg0']:.6f}")
                
                st.markdown("### 📊 Training Data")
                st.dataframe(training_data[['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 
                                         'metrics/mAP50(B)', 'metrics/mAP50-95(B)']], 
                            use_container_width=True)
            
            with train_col2:
                st.markdown("### 📊 Training Metrics Over Time")
                fig5, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(12, 8))
                
                # Loss curves
                ax5.plot(training_data['epoch'], training_data['train/box_loss'], 'b-', label='Train Box Loss')
                ax5.plot(training_data['epoch'], training_data['val/box_loss'], 'r-', label='Val Box Loss')
                ax5.set_xlabel('Epoch')
                ax5.set_ylabel('Box Loss')
                ax5.set_title('Box Loss')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                
                ax6.plot(training_data['epoch'], training_data['train/cls_loss'], 'b-', label='Train Class Loss')
                ax6.plot(training_data['epoch'], training_data['val/cls_loss'], 'r-', label='Val Class Loss')
                ax6.set_xlabel('Epoch')
                ax6.set_ylabel('Classification Loss')
                ax6.set_title('Classification Loss')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
                
                # Metrics
                ax7.plot(training_data['epoch'], training_data['metrics/precision(B)'], 'g-', label='Precision')
                ax7.plot(training_data['epoch'], training_data['metrics/recall(B)'], 'orange', label='Recall')
                ax7.set_xlabel('Epoch')
                ax7.set_ylabel('Score')
                ax7.set_title('Precision & Recall')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
                
                ax8.plot(training_data['epoch'], training_data['metrics/mAP50(B)'], 'purple', label='mAP@0.5')
                ax8.plot(training_data['epoch'], training_data['metrics/mAP50-95(B)'], 'brown', label='mAP@0.5:0.95')
                ax8.set_xlabel('Epoch')
                ax8.set_ylabel('mAP Score')
                ax8.set_title('Mean Average Precision')
                ax8.legend()
                ax8.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig5)

else:
    st.info("Please upload one or more lunar surface images to begin multi-image detection analysis.")
    
    # Show sample information when no images are uploaded
    st.markdown("---")
    st.markdown("## 🚀 Multi-Image Analysis Features")
    st.markdown("""
    This enhanced lunar surface analysis tool now supports **batch processing** of multiple images with:
    
    **🔄 Batch Processing:**
    - Upload multiple images simultaneously
    - Progress tracking during processing
    - Parallel analysis capabilities
    
    **🖼️ Prioritized Output Display:**
    - **Detection results shown first** - immediate visual feedback
    - Quick summary metrics for overview
    - Grid layout for easy comparison
    
    **📊 Detailed Analysis (Expandable):**
    - Comprehensive statistical comparisons
    - Correlation analysis between features
    - Performance visualization charts
    - Export-ready comparison tables
    
    **🔍 Individual Comparisons:**
    - Original vs processed side-by-side views
    - Selectable image display options
    - Individual performance metrics
    
    **⚙️ Model Training Insights:**
    - Training performance visualization
    - Loss curves and metrics over time
    - Final model performance summary
    
    **✨ Enhanced User Experience:**
    - Results displayed immediately after processing
    - Detailed analysis available in expandable sections
    - Organized information hierarchy for better navigation
    """)
