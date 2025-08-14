import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self):
        """Initialize the DataCleaner with empty dataframes"""
        self.categoria_df = None
        self.cliente_df = None
        self.events_df = None
        self.marca_df = None
        self.producto_df = None
        self.cleaning_report = []
    
    def inspect_data(self):
        """Quick inspection of loaded data to understand formats"""
        print("\n--- Data Inspection ---")
        if self.events_df is not None:
            print("\nEvents dataset sample:")
            print(f"  Timestamp dtype: {self.events_df['timestamp'].dtype}")
            print(f"  Timestamp sample values: {self.events_df['timestamp'].head(3).tolist()}")
            if self.events_df['timestamp'].dtype in ['float64', 'int64']:
                print(f"  Timestamp range: {self.events_df['timestamp'].min()} to {self.events_df['timestamp'].max()}")
    
    def load_data(self):
        """Load all CSV files with appropriate encoding"""
        print("Loading datasets...")
        
        # Load categoria.csv (UTF-8)
        self.categoria_df = pd.read_csv('categoria.csv', encoding='utf-8')
        print(f"✓ Loaded categoria.csv: {self.categoria_df.shape}")
        
        # Load cliente.csv (ISO-8859-1)
        self.cliente_df = pd.read_csv('cliente.csv', encoding='iso-8859-1')
        print(f"✓ Loaded cliente.csv: {self.cliente_df.shape}")
        
        # Load events.csv (UTF-8)
        self.events_df = pd.read_csv('events.csv', encoding='utf-8')
        print(f"✓ Loaded events.csv: {self.events_df.shape}")
        
        # Load marca.csv (UTF-8)
        self.marca_df = pd.read_csv('marca.csv', encoding='utf-8')
        print(f"✓ Loaded marca.csv: {self.marca_df.shape}")
        
        # Load producto.csv (UTF-8)
        self.producto_df = pd.read_csv('producto.csv', encoding='utf-8')
        print(f"✓ Loaded producto.csv: {self.producto_df.shape}")
        
    def clean_categoria(self):
        """Clean categoria dataset"""
        print("\n--- Cleaning categoria.csv ---")
        initial_shape = self.categoria_df.shape
        
        # Remove duplicates based on id
        self.categoria_df = self.categoria_df.drop_duplicates(subset=['id'])
        
        # Remove duplicates based on categoria name (case-insensitive)
        self.categoria_df['categoria_lower'] = self.categoria_df['categoria'].str.lower().str.strip()
        self.categoria_df = self.categoria_df.drop_duplicates(subset=['categoria_lower'])
        self.categoria_df = self.categoria_df.drop('categoria_lower', axis=1)
        
        # Clean text: trim whitespace, standardize capitalization
        self.categoria_df['categoria'] = self.categoria_df['categoria'].str.strip()
        self.categoria_df['categoria'] = self.categoria_df['categoria'].str.title()
        
        # Remove rows with null values
        self.categoria_df = self.categoria_df.dropna()
        
        # Ensure id is integer
        self.categoria_df['id'] = self.categoria_df['id'].astype(int)
        
        rows_removed = initial_shape[0] - self.categoria_df.shape[0]
        self.cleaning_report.append(f"Categoria: Removed {rows_removed} rows")
        print(f"  Removed {rows_removed} rows")
        
    def clean_marca(self):
        """Clean marca dataset"""
        print("\n--- Cleaning marca.csv ---")
        initial_shape = self.marca_df.shape
        
        # Remove duplicates based on id
        self.marca_df = self.marca_df.drop_duplicates(subset=['id'])
        
        # Remove duplicates based on marca name (case-insensitive)
        self.marca_df['marca_lower'] = self.marca_df['marca'].str.lower().str.strip()
        self.marca_df = self.marca_df.drop_duplicates(subset=['marca_lower'])
        self.marca_df = self.marca_df.drop('marca_lower', axis=1)
        
        # Clean text: trim whitespace, standardize capitalization
        self.marca_df['marca'] = self.marca_df['marca'].str.strip()
        self.marca_df['marca'] = self.marca_df['marca'].str.title()
        
        # Remove rows with null values
        self.marca_df = self.marca_df.dropna()
        
        # Ensure id is integer
        self.marca_df['id'] = self.marca_df['id'].astype(int)
        
        rows_removed = initial_shape[0] - self.marca_df.shape[0]
        self.cleaning_report.append(f"Marca: Removed {rows_removed} rows")
        print(f"  Removed {rows_removed} rows")
        
    def clean_cliente(self):
        """Clean cliente dataset"""
        print("\n--- Cleaning cliente.csv ---")
        initial_shape = self.cliente_df.shape
        
        # Remove complete duplicates
        self.cliente_df = self.cliente_df.drop_duplicates()
        
        # Convert id to integer (handle NaN first)
        self.cliente_df = self.cliente_df.dropna(subset=['id'])
        self.cliente_df['id'] = self.cliente_df['id'].astype(int)
        
        # Clean text fields
        text_columns = ['nombre', 'apellido', 'empresa', 'idioma', 'puesto', 'ciudad']
        for col in text_columns:
            if col in self.cliente_df.columns:
                self.cliente_df[col] = self.cliente_df[col].str.strip()
                # Capitalize names properly
                if col in ['nombre', 'apellido', 'ciudad']:
                    self.cliente_df[col] = self.cliente_df[col].str.title()
        
        # Clean and validate email
        self.cliente_df['correo'] = self.cliente_df['correo'].str.strip().str.lower()
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        invalid_emails = ~self.cliente_df['correo'].str.match(email_pattern, na=False)
        print(f"  Found {invalid_emails.sum()} invalid emails")
        
        # Clean NIT (remove special characters, keep only alphanumeric)
        self.cliente_df['nit'] = self.cliente_df['nit'].str.replace(r'[^A-Za-z0-9]', '', regex=True)
        
        # Clean phone numbers (remove special characters)
        self.cliente_df['telefono'] = self.cliente_df['telefono'].str.replace(r'[^\d+]', '', regex=True)
        
        # Standardize gender values
        self.cliente_df['genero'] = self.cliente_df['genero'].str.strip().str.upper()
        self.cliente_df['genero'] = self.cliente_df['genero'].replace({
            'M': 'M', 'MASCULINO': 'M', 'MALE': 'M', 'HOMBRE': 'M',
            'F': 'F', 'FEMENINO': 'F', 'FEMALE': 'F', 'MUJER': 'F'
        })
        
        # Parse and validate birth dates
        try:
            self.cliente_df['nacimiento'] = pd.to_datetime(self.cliente_df['nacimiento'], errors='coerce')
            # Remove invalid dates (future dates or too old)
            current_year = datetime.now().year
            self.cliente_df = self.cliente_df[
                (self.cliente_df['nacimiento'].dt.year <= current_year) & 
                (self.cliente_df['nacimiento'].dt.year >= 1900)
            ]
        except:
            print("  Warning: Could not parse birth dates")
        
        # Remove rows with critical missing values
        critical_columns = ['id', 'nombre', 'apellido']
        self.cliente_df = self.cliente_df.dropna(subset=critical_columns)
        
        rows_removed = initial_shape[0] - self.cliente_df.shape[0]
        self.cleaning_report.append(f"Cliente: Removed {rows_removed} rows")
        print(f"  Removed {rows_removed} rows")
        
    def clean_producto(self):
        """Clean producto dataset"""
        print("\n--- Cleaning producto.csv ---")
        initial_shape = self.producto_df.shape
        
        # Remove duplicates based on id
        self.producto_df = self.producto_df.drop_duplicates(subset=['id'])
        
        # Ensure correct data types
        self.producto_df['id'] = self.producto_df['id'].astype(int)
        self.producto_df['volumen'] = self.producto_df['volumen'].astype(int)
        
        # Clean product names
        self.producto_df['nombre'] = self.producto_df['nombre'].str.strip()
        
        # Validate foreign keys
        valid_categoria_ids = set(self.categoria_df['id'].values)
        valid_marca_ids = set(self.marca_df['id'].values)
        
        # Check for invalid categoria_id
        invalid_categoria = ~self.producto_df['categoria_id'].isin(valid_categoria_ids) & self.producto_df['categoria_id'].notna()
        if invalid_categoria.sum() > 0:
            print(f"  Found {invalid_categoria.sum()} products with invalid categoria_id")
            # Option: remove or set to null
            self.producto_df.loc[invalid_categoria, 'categoria_id'] = np.nan
        
        # Check for invalid marca_id
        invalid_marca = ~self.producto_df['marca_id'].isin(valid_marca_ids) & self.producto_df['marca_id'].notna()
        if invalid_marca.sum() > 0:
            print(f"  Found {invalid_marca.sum()} products with invalid marca_id")
            # Option: remove or set to null
            self.producto_df.loc[invalid_marca, 'marca_id'] = np.nan
        
        # Convert foreign keys to nullable integers
        self.producto_df['categoria_id'] = self.producto_df['categoria_id'].astype('Int64')
        self.producto_df['marca_id'] = self.producto_df['marca_id'].astype('Int64')
        
        # Validate price (should be positive)
        invalid_price = self.producto_df['precio'] <= 0
        if invalid_price.sum() > 0:
            print(f"  Found {invalid_price.sum()} products with invalid prices")
            self.producto_df = self.producto_df[~invalid_price]
        
        # Validate volume (should be positive)
        invalid_volume = self.producto_df['volumen'] <= 0
        if invalid_volume.sum() > 0:
            print(f"  Found {invalid_volume.sum()} products with invalid volumes")
            self.producto_df = self.producto_df[~invalid_volume]
        
        # Remove rows with missing critical values
        critical_columns = ['id', 'nombre', 'precio']
        self.producto_df = self.producto_df.dropna(subset=critical_columns)
        
        rows_removed = initial_shape[0] - self.producto_df.shape[0]
        self.cleaning_report.append(f"Producto: Removed {rows_removed} rows")
        print(f"  Removed {rows_removed} rows")
        
    def clean_events(self):
        """Clean events dataset"""
        print("\n--- Cleaning events.csv ---")
        initial_shape = self.events_df.shape
        
        # Remove complete duplicates
        self.events_df = self.events_df.drop_duplicates()
        
        # Handle timestamp conversion more carefully
        if self.events_df['timestamp'].dtype in ['float64', 'int64']:
            # Check the range of timestamp values to determine the unit
            timestamp_max = self.events_df['timestamp'].max()
            timestamp_min = self.events_df['timestamp'].min()
            
            print(f"  Timestamp range: {timestamp_min} to {timestamp_max}")
            
            # Check if timestamps are reasonable Unix timestamps
            # Unix timestamp for year 2000: ~946684800
            # Unix timestamp for year 2030: ~1893456000
            # Unix timestamp in ms for year 2000: ~946684800000
            # Unix timestamp in ms for year 2030: ~1893456000000
            
            try:
                if timestamp_min > 1e15:  # Extremely large values
                    print("  Warning: Timestamp values are extremely large")
                    # These might be nanoseconds or corrupted data
                    # Try to scale them down or keep as is
                    if timestamp_max < 1e18:
                        try:
                            # Try nanoseconds
                            self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'], unit='ns')
                            print("  Converted timestamps from nanoseconds")
                        except:
                            # Scale down to a reasonable range
                            print("  Scaling down timestamp values")
                            scale_factor = 1e9
                            self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'] / scale_factor, unit='s')
                    else:
                        print("  Keeping timestamps as numeric values due to extreme values")
                elif timestamp_max > 1e12:  # Likely milliseconds
                    print("  Detected timestamps in milliseconds")
                    self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'], unit='ms')
                elif timestamp_max > 1e10:  # Likely microseconds
                    print("  Detected timestamps in microseconds")
                    self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'], unit='us')
                elif timestamp_max > 1e7:  # Likely seconds (reasonable range)
                    print("  Detected timestamps in seconds")
                    self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'], unit='s')
                else:
                    # Very small values, might be days or hours since epoch
                    print("  Timestamp values are small, might be days since epoch")
                    try:
                        self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'], unit='D')
                    except:
                        print("  Keeping timestamps as numeric values")
            except Exception as e:
                print(f"  Warning: Could not convert timestamps to datetime: {e}")
                print("  Keeping timestamps as numeric values")
        elif self.events_df['timestamp'].dtype == 'object':
            # Try to parse as string datetime
            try:
                self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'])
            except:
                print("  Warning: Could not parse timestamp strings")
        
        # Ensure correct data types for other columns
        self.events_df['visitorid'] = self.events_df['visitorid'].astype(int)
        self.events_df['itemid'] = self.events_df['itemid'].astype(int)
        
        # Clean event types
        self.events_df['event'] = self.events_df['event'].str.strip().str.lower()
        
        # Standardize event names
        event_mapping = {
            'view': 'view',
            'views': 'view',
            'cart': 'addtocart',
            'add_to_cart': 'addtocart',
            'addtocart': 'addtocart',
            'transaction': 'transaction',
            'purchase': 'transaction',
            'buy': 'transaction'
        }
        self.events_df['event'] = self.events_df['event'].replace(event_mapping)
        
        # Validate itemid against products (only if producto_df has been cleaned)
        if self.producto_df is not None and len(self.producto_df) > 0:
            valid_product_ids = set(self.producto_df['id'].values)
            invalid_items = ~self.events_df['itemid'].isin(valid_product_ids)
            if invalid_items.sum() > 0:
                print(f"  Found {invalid_items.sum()} events with invalid itemid")
                self.events_df = self.events_df[~invalid_items]
        else:
            print("  Warning: Cannot validate itemid against products")
        
        # Convert transactionid to nullable integer
        self.events_df['transactionid'] = self.events_df['transactionid'].astype('Int64')
        
        # Validate: transactions should have transactionid
        transaction_events = self.events_df['event'] == 'transaction'
        missing_transaction_id = transaction_events & self.events_df['transactionid'].isna()
        if missing_transaction_id.sum() > 0:
            print(f"  Found {missing_transaction_id.sum()} transaction events without transaction ID")
            self.events_df = self.events_df[~missing_transaction_id]
        
        # Remove rows with missing critical values
        critical_columns = ['timestamp', 'visitorid', 'event', 'itemid']
        self.events_df = self.events_df.dropna(subset=critical_columns)
        
        # Sort by timestamp (whether datetime or numeric)
        try:
            self.events_df = self.events_df.sort_values('timestamp')
        except:
            print("  Warning: Could not sort by timestamp")
        
        rows_removed = initial_shape[0] - self.events_df.shape[0]
        self.cleaning_report.append(f"Events: Removed {rows_removed} rows")
        print(f"  Removed {rows_removed} rows")
        
    def validate_data_integrity(self):
        """Perform cross-table validation and integrity checks"""
        print("\n--- Validating Data Integrity ---")
        
        # Check for orphaned records
        print("  Checking for orphaned records...")
        
        # Products without valid categoria
        orphaned_products_cat = self.producto_df['categoria_id'].isna().sum()
        print(f"    Products without categoria: {orphaned_products_cat}")
        
        # Products without valid marca
        orphaned_products_marca = self.producto_df['marca_id'].isna().sum()
        print(f"    Products without marca: {orphaned_products_marca}")
        
        # Events referencing non-existent products (should be 0 after cleaning)
        product_ids = set(self.producto_df['id'].values)
        orphaned_events = ~self.events_df['itemid'].isin(product_ids)
        print(f"    Events with invalid products: {orphaned_events.sum()}")
        
    def generate_summary_report(self):
        """Generate a summary report of the cleaning process"""
        print("\n" + "="*50)
        print("CLEANING SUMMARY REPORT")
        print("="*50)
        
        print("\nFinal Dataset Shapes:")
        print(f"  categoria.csv: {self.categoria_df.shape}")
        print(f"  cliente.csv: {self.cliente_df.shape}")
        print(f"  events.csv: {self.events_df.shape}")
        print(f"  marca.csv: {self.marca_df.shape}")
        print(f"  producto.csv: {self.producto_df.shape}")
        
        print("\nCleaning Actions:")
        for action in self.cleaning_report:
            print(f"  {action}")
        
        print("\nData Quality Metrics:")
        print(f"  Unique categories: {self.categoria_df['id'].nunique()}")
        print(f"  Unique brands: {self.marca_df['id'].nunique()}")
        print(f"  Unique products: {self.producto_df['id'].nunique()}")
        print(f"  Unique customers: {self.cliente_df['id'].nunique()}")
        print(f"  Unique visitors in events: {self.events_df['visitorid'].nunique()}")
        print(f"  Total events: {len(self.events_df)}")
        print(f"  Event types: {self.events_df['event'].value_counts().to_dict()}")
        
    def save_cleaned_data(self, output_dir='cleaned_data'):
        """Save cleaned datasets to CSV files"""
        import os
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n--- Saving cleaned data to {output_dir}/ ---")
        
        # Save all cleaned datasets
        self.categoria_df.to_csv(f'{output_dir}/categoria_cleaned.csv', index=False, encoding='utf-8')
        print(f"  ✓ Saved categoria_cleaned.csv")
        
        self.cliente_df.to_csv(f'{output_dir}/cliente_cleaned.csv', index=False, encoding='utf-8')
        print(f"  ✓ Saved cliente_cleaned.csv")
        
        self.events_df.to_csv(f'{output_dir}/events_cleaned.csv', index=False, encoding='utf-8')
        print(f"  ✓ Saved events_cleaned.csv")
        
        self.marca_df.to_csv(f'{output_dir}/marca_cleaned.csv', index=False, encoding='utf-8')
        print(f"  ✓ Saved marca_cleaned.csv")
        
        self.producto_df.to_csv(f'{output_dir}/producto_cleaned.csv', index=False, encoding='utf-8')
        print(f"  ✓ Saved producto_cleaned.csv")
        
        # Save cleaning report
        with open(f'{output_dir}/cleaning_report.txt', 'w') as f:
            f.write("DATA CLEANING REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Final Dataset Shapes:\n")
            f.write(f"  categoria.csv: {self.categoria_df.shape}\n")
            f.write(f"  cliente.csv: {self.cliente_df.shape}\n")
            f.write(f"  events.csv: {self.events_df.shape}\n")
            f.write(f"  marca.csv: {self.marca_df.shape}\n")
            f.write(f"  producto.csv: {self.producto_df.shape}\n\n")
            
            f.write("Cleaning Actions:\n")
            for action in self.cleaning_report:
                f.write(f"  {action}\n")
        
        print(f"  ✓ Saved cleaning_report.txt")
        
    def run_full_cleaning_pipeline(self, inspect_first=True):
        """Execute the complete data cleaning pipeline"""
        print("="*50)
        print("STARTING DATA CLEANING PIPELINE")
        print("="*50)
        
        # Load all data
        self.load_data()
        
        # Optional: Inspect data before cleaning
        if inspect_first:
            self.inspect_data()
        
        # Clean each dataset
        self.clean_categoria()
        self.clean_marca()
        self.clean_cliente()
        self.clean_producto()
        self.clean_events()
        
        # Validate data integrity
        self.validate_data_integrity()
        
        # Generate summary report
        self.generate_summary_report()
        
        # Save cleaned data
        self.save_cleaned_data()
        
        print("\n" + "="*50)
        print("DATA CLEANING COMPLETED SUCCESSFULLY!")
        print("="*50)


# Main execution
if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run_full_cleaning_pipeline()
    
    # Optional: Return cleaned dataframes for further analysis
    # You can access the cleaned data through:
    # cleaner.categoria_df
    # cleaner.cliente_df
    # cleaner.events_df
    # cleaner.marca_df
    # cleaner.producto_df