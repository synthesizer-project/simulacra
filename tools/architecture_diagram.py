"""
This script contains the Mermaid syntax for the project's network architecture diagram.
It can be used to regenerate the diagram or for documentation purposes.
"""

DIAGRAM_DEFINITION = """
graph TD
    InputParams["Input<br/>(Age, Metallicity)"] -- "Inference Path" --> Regressor["Regressor MLP"]
    InputSpectrum["Input<br/>Spectrum"] -- "AE Training Path" --> Encoder["Encoder MLP"]

    Regressor --> LatentSpace{"Shared<br/>Latent Space"}
    Encoder --> LatentSpace

    LatentSpace --> Decoder["Decoder MLP"]
    Decoder --> OutputSpectrum["Output<br/>Spectrum"]

    subgraph Autoencoder
        Encoder
        Decoder
    end

    style Regressor fill:#D5E8D4,stroke:#82B366,stroke-width:2px
    style Autoencoder fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px
    style LatentSpace fill:#FFE6CC,stroke:#D79B00,stroke-width:2px
"""

def save_diagram_definition(filepath="figures/network_diagram.md"):
    """Saves the diagram definition to a file."""
    try:
        with open(filepath, 'w') as f:
            f.write("```mermaid\\n")
            f.write(DIAGRAM_DEFINITION)
            f.write("\\n```")
        print(f"Diagram definition saved to {filepath}")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)
    save_diagram_definition() 