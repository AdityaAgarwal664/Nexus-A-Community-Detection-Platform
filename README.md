# Nexus: Community Detection Visualization Platform

Nexus is a dynamic and interactive platform designed to detect and visualize communities within networks. It provides users with various algorithms to identify influencers and sort them based on community density or size. The platform is built using Python, Flask, HTML, CSS, JavaScript, and NetworkX.

## Key Features

- **Influencer Detection**: Specify the required number of influencers within each community.
- **Algorithm Selection**: Choose from multiple community detection algorithms:
  - Louvain
  - Max-Min
  - ABC
  - Label Propagation
- **Community Sorting**: Sort influencers per community based on:
  - Community Density
  - Community Size
- **Visualization**: Comprehensive visualization of communities and influencers.

## Demo

Check out the [demo video](https://youtu.be/1ug51hOdr7w) to see Nexus in action!

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Graph Analysis**: NetworkX

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Nexus.git
    cd Nexus
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Flask app:
    ```bash
    flask run
    ```

5. Open your browser and go to `http://127.0.0.1:5000` to access Nexus.

## Usage

1. **Upload Network Data**: Upload your network data in the required format.
2. **Specify Influencers**: Enter the number of influencers you want to identify per community.
3. **Select Algorithm**: Choose one of the community detection algorithms.
4. **Sort Influencers**: Decide whether to sort influencers based on community density or size.
5. **Visualize**: View the detected communities and influencers on the interactive visualization panel.

## Project Structure

```
Nexus/
├── static/
│   ├── css/
│   ├── js/
│   └── ...
├── templates/
│   ├── index.html
│   └── ...
├── app.py
├── requirements.txt
└── README.md
```

- **static/**: Contains static files (CSS, JavaScript).
- **templates/**: Contains HTML templates.
- **app.py**: The main Flask application file.
- **requirements.txt**: List of dependencies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or questions, please contact [your-email@example.com].

---

Thank you for using Nexus! We hope it serves your community detection and visualization needs effectively.
