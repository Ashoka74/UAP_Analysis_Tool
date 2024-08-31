# UAP Analysis Software: A Comprehensive Tool for Exploratory Data Analysis of Unidentified Anomalous Phenomena

Our tool aims to standardize UAP data analysis methodologies and facilitate collaborative research in the scientific community. By providing a user-friendly interface for complex data processing and analysis tasks, it enables researchers to focus on interpreting results and drawing insights from UAP data.


[Explore Analysis Tool Here](https://huggingface.co/organizations/UFOSINT/share/NtOisEiDJyzNmVuZYXTTkZwpelsWlrqOTs)</h3>


## Key Features

1. **Feature Parsing with Large Language Models (LLMs)**
   - Parse relevant information from unstructured reports into structured data.
   - Customizable JSON templates for tailored parsing.
   - Enables structured data analysis with classic statistical methods.
  
https://github.com/Ashoka74/UAP_Analysis_Tool/assets/40150735/a122d9da-e9e0-4475-8a0e-63d7f6e04785

<video width="600" controls>
  <source src="https://youtu.be/AaNEORiyyA8?si=w-4TFW-vvccnH7uD" type="video/mp4">
  Your browser does not support the video tag.
</video>

2. **Semantic Search and Summarization**
- Implements semantic search across multiple columns using natural language queries.
- Ranks and sorts dataset based on query relevance.
- Summarize and query multiple reports

![SemSearch](IMG-20240718-WA0001.jpg?raw=true "Semantic Search")

![Parsing, Filtering and Querying](UFO_APP_SCREENSHOT.png?raw=true "Parsing, Filtering and Querying")


3. **Interactive Data Filtering and Visualization**
   - User-friendly interface for applying multiple filters to the dataset.
   - Dynamic visualization of filtered data using various plot types (treemaps, histograms, line plots, bar charts).
   - Supports both categorical and numerical data analysis.
   - Creates interactive visualizations for cluster analysis and feature correlations.
  
![Dynamic Visualizations](IMG-20240718-WA0003.jpg?raw=true "Dynamic Visualizations")
</br>

4. **Magnetic Data Analysis**
   - Retrieves magnetic field data from INTERMAGNET geomagnetic stations.
   - Correlates magnetic data with UAP sighting times and locations.
   - Implements FastDTW (Dynamic Time Warping) for aligning potential anomalous geomagnetic signatures across multiple sightings.
   - Visualizes individual and aggregated magnetic data for analysis.
  
   ![custom_plot_TUC_2023-10-27T18_46_00](https://github.com/user-attachments/assets/eb13939e-a186-4ac8-895e-82e528be534b)


5. **Geospatial Visualization**
   - Interactive map interface for visualizing UAP sightings.
   - Incorporates layers for military bases, nuclear power plants, and UAP sightings.
   - Allows filtering and updating of map data based on various attributes.
    
![Dynamic Map](IMG-20240718-WA0000.jpg?raw=true "Dynamic Map")


6. **Data Processing and Analysis Pipeline**
   - Utilizes UMAP and HDBSCAN for clustering UAP reports and visualizing clusters.
   - Employs XGBoost for feature importance analysis in UAP classifications.
   - Performs statistical analyses including V-Cramer correlation and chi-square tests.
   - Implements parallel processing for improved performance on large datasets.
  
![Variable Correlations](corr_grouped_features.png?raw=true "Correlation of Variables")</br>

![Analysis of Variable importances](Analyzer_additionalInformation.interactionWithEnvironment_0.75_prediction_XGB.jpeg?raw=true "Variable weights for decision-tree classification")


  


## Contributing

Please feel free to contribute to this project by submitting a pull request or opening an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
