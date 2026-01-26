"""
Example: Platform-aware CZI loading for napari-czitools plugin.

This module shows how to handle Linux threading issues when czitools
is used as part of a Napari plugin.
"""

import platform
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from czitools.read_tools import read_tools
from czitools.utils.planetable import get_planetable
from czitools.metadata_tools.czi_metadata import CziMetadata


class NapariCziLoader:
    """
    Platform-aware CZI loader for Napari plugins.
    
    Handles threading issues on Linux automatically.
    """
    
    def __init__(self, enable_planetable_on_linux: bool = False):
        """
        Initialize loader.
        
        Args:
            enable_planetable_on_linux: If True, attempt planetable extraction
                on Linux (may crash). If False, skip planetable on Linux.
        """
        self.enable_planetable_on_linux = enable_planetable_on_linux
        self.is_linux = platform.system() == "Linux"
    
    def load_czi(
        self,
        filepath: Path,
        extract_planetable: bool = True
    ) -> Tuple[np.ndarray, CziMetadata, Optional[pd.DataFrame]]:
        """
        Load CZI file with platform-aware handling.
        
        Args:
            filepath: Path to CZI file
            extract_planetable: Whether to attempt planetable extraction
            
        Returns:
            Tuple of (array, metadata, planetable_df)
            planetable_df may be None on Linux or if extraction fails
        """
        print(f"Loading CZI: {filepath}")
        
        # Always load image data (thread-safe)
        array, metadata = read_tools.read_6darray(
            filepath,
            use_dask=True,
            use_xarray=True,
            chunk_zyx=True
        )
        print(f"✅ Image loaded: {array.shape}")
        
        # Handle planetable based on platform
        planetable_df = None
        
        if extract_planetable:
            if self.is_linux and not self.enable_planetable_on_linux:
                # Skip on Linux by default
                print("ℹ️  Planetable extraction disabled on Linux (threading safety)")
                print("   To enable: set enable_planetable_on_linux=True")
                print("   Warning: May cause crashes with Napari on Linux")
            else:
                # Try extraction
                planetable_df = self._extract_planetable_safe(filepath)
        
        return array, metadata, planetable_df
    
    def _extract_planetable_safe(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Safely attempt planetable extraction with error handling.
        
        Args:
            filepath: Path to CZI file
            
        Returns:
            DataFrame or None if extraction fails
        """
        if self.is_linux:
            print("⚠️  WARNING: Attempting planetable extraction on Linux")
            print("   This may cause Napari to crash due to threading conflicts")
            print("   If crashes occur, set CZITOOLS_DISABLE_AICSPYLIBCZI=1")
        
        try:
            df, _ = get_planetable(filepath, norm_time=True)
            print(f"✅ Planetable extracted: {len(df)} rows")
            return df
            
        except RuntimeError as e:
            if "CZITOOLS_DISABLE_AICSPYLIBCZI" in str(e):
                print("ℹ️  Planetable disabled (safe mode active)")
            else:
                print(f"❌ Planetable extraction failed: {e}")
            return None
            
        except Exception as e:
            print(f"❌ Planetable extraction error: {e}")
            if self.is_linux:
                print("   This may be a threading conflict on Linux")
                print("   Restart Napari with: export CZITOOLS_DISABLE_AICSPYLIBCZI=1")
            return None


# Example usage in a napari plugin widget
def example_napari_plugin_widget():
    """
    Example of how to use NapariCziLoader in a napari plugin.
    """
    from magicgui import magic_factory
    from napari.types import LayerDataTuple
    
    @magic_factory(
        call_button="Load CZI",
        filepath={"mode": "r", "filter": "*.czi"},
        enable_planetable={"label": "Extract Planetable (may crash on Linux)"}
    )
    def load_czi_widget(
        filepath: Path,
        enable_planetable: bool = False
    ) -> LayerDataTuple:
        """
        Napari widget to load CZI files.
        
        Args:
            filepath: CZI file to load
            enable_planetable: Extract planetable (risky on Linux)
            
        Returns:
            Layer data tuple for Napari
        """
        # Create loader with platform awareness
        loader = NapariCziLoader(
            enable_planetable_on_linux=enable_planetable
        )
        
        # Load CZI
        array, metadata, planetable_df = loader.load_czi(
            filepath,
            extract_planetable=enable_planetable
        )
        
        # Prepare metadata for Napari
        layer_metadata = {
            'czi_metadata': metadata.info,
            'planetable': planetable_df.to_dict() if planetable_df is not None else None
        }
        
        # Show planetable summary if available
        if planetable_df is not None:
            print(f"\n📊 Planetable Summary:")
            print(f"   Total planes: {len(planetable_df)}")
            if 'Time[s]' in planetable_df.columns:
                print(f"   Time range: {planetable_df['Time[s]'].min():.2f} - {planetable_df['Time[s]'].max():.2f} s")
        
        # Return layer data tuple
        return (array, {'name': filepath.name, 'metadata': layer_metadata}, 'image')
    
    return load_czi_widget


# Example: Manual usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Create loader
    loader = NapariCziLoader(
        enable_planetable_on_linux=False  # Safe default for Linux
    )
    
    # Load CZI
    filepath = Path("data/CellDivision_T3_Z5_CH2_X240_Y170.czi")
    array, metadata, planetable_df = loader.load_czi(filepath)
    
    print(f"\n✅ Loaded successfully:")
    print(f"   Shape: {array.shape}")
    print(f"   Planetable: {'Available' if planetable_df is not None else 'Not extracted'}")
    
    # If running in Napari, could display now:
    # import napari
    # viewer = napari.current_viewer()
    # viewer.add_image(array, name=filepath.name)
