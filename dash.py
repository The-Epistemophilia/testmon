import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from datetime import datetime


# Step 1: Configuration Class (Same as before)
class Config:
    COLORS = {
        'primary': '#3498db',
        'secondary': '#2c3e50',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'light': '#ecf0f1',
        'dark': '#34495e'
    }
    CHART_TEMPLATE = 'plotly_white'
    PLATFORMS = {
        'SnappPay': {'color': '#3498db', 'icon': '🏪'},
        'Torob': {'color': '#27ae60', 'icon': '🔍'},
    }


# Step 2: Data Processor Class (Same as before)
class DataProcessor:
    def __init__(self, file_path=None):
        self.main_df = None
        self.product_index = {}
        self.load_main_data(file_path)

    def load_main_data(self, file_path):
        try:
            # if file_path and os.path.exists(file_path):
            if file_path:
                exclude_list = ['mon2001638', 'mon2006582', 'mon2003801', 'mon2003803', 'mon1122162', 'mon6611',
                                'mon2006589', 'mon2000093', 'mon2000088'
                    , 'mon2009043', 'mon2006662', 'mon2003796', 'mon2000006', 'mon2001123', 'mon2006811', 'mon2006594']

                st.write("📁 Loading main product pool...")
                self.main_df = pd.read_csv(
                    file_path,
                    low_memory=False,
                )
                print('=====================================================================')
                print(self.main_df.info())
                print(self.main_df.columns)
                print('=====================================================================')
                # Convert prices
                self.main_df['Product Prices'] = pd.to_numeric(
                    self.main_df['Product Prices'],
                    errors='coerce'
                )

                # Drop NA prices
                self.main_df = self.main_df.dropna(subset=['Product Prices'])

                # 🔥 Exclude unwanted MON values
                self.main_df = self.main_df[~self.main_df['MON'].isin(exclude_list)]

                st.write(f"✅ Main pool loaded: {len(self.main_df):,} records")

            else:
                st.write("📁 Creating sample data...")
                self.main_df = self.create_sample_data()

            self._build_product_index()

        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            self.main_df = self.create_sample_data()
            self._build_product_index()

    def _build_product_index(self):
        st.write("🔨 Building product index...")
        self.product_index = {}

        for mon, group in self.main_df.groupby('MON'):
            if not group.empty:
                product_title = group['Product Titles'].iloc[0] if pd.notna(
                    group['Product Titles'].iloc[0]) else "Unknown"
                seller_count = group['Seller'].nunique()

                platform_counts = group['Platform'].value_counts().to_dict()
                platforms_present = group['Platform'].unique().tolist()

                platform_availability = {}
                for platform in Config.PLATFORMS.keys():
                    platform_availability[platform] = platform in platform_counts

                self.product_index[mon] = {
                    'title': str(product_title),
                    'seller_count': seller_count,
                    'platforms': platforms_present,
                    'platform_counts': platform_counts,
                    'platform_availability': platform_availability,
                    'cross_platform': len(platforms_present) > 1
                }

        st.write(f"✅ Product index built: {len(self.product_index)} unique products")

    def search_products(self, search_terms, sort_order='ascending'):
        """Search products with sort order option"""
        if not search_terms or not search_terms.strip():
            products = list(self.product_index.keys())
        else:
            search_terms = [term.strip().lower() for term in search_terms.split() if term.strip()]
            products = []

            for mon, product_info in self.product_index.items():
                title_lower = product_info['title'].lower()
                mon_lower = str(mon).lower()

                if (all(term in title_lower for term in search_terms) or
                        all(term in mon_lower for term in search_terms)):
                    products.append(mon)

        # Sort by seller count
        reverse = (sort_order == 'descending')
        return sorted(products, key=lambda x: self.product_index[x]['seller_count'], reverse=reverse)

    def get_product_data(self, mon_value):
        if mon_value not in self.product_index:
            return pd.DataFrame()

        product_data = self.main_df[self.main_df['MON'] == mon_value].copy()
        product_data['Product Prices'] = pd.to_numeric(product_data['Product Prices'], errors='coerce')
        product_data = product_data.dropna(subset=['Product Prices'])

        if product_data.empty:
            return pd.DataFrame()

        return self._add_analytical_columns(product_data)

    def _add_analytical_columns(self, df):
        df_analytical = df.copy()

        try:
            product_stats = df_analytical.groupby('MON')['Product Prices'].agg([
                'min', 'max', 'mean', 'std', 'count'
            ]).round(2).reset_index()
            product_stats.columns = ['MON', 'min_price', 'max_price', 'mean_price', 'std_price', 'seller_count']

            df_analytical = df_analytical.merge(product_stats, on='MON', how='left')

            df_analytical['price_vs_avg'] = df_analytical['Product Prices'] - df_analytical['mean_price']
            df_analytical['savings_opportunity'] = df_analytical['Product Prices'] - df_analytical['min_price']
            df_analytical['is_cheapest'] = df_analytical['Product Prices'] == df_analytical['min_price']
            df_analytical['price_ratio'] = ((df_analytical['Product Prices'] / df_analytical['mean_price']) - 1) * 100

            conditions = [
                df_analytical['is_cheapest'],
                df_analytical['Product Prices'] <= df_analytical['mean_price']
            ]
            choices = ['Cheapest', 'Below Average']
            df_analytical['price_segment'] = np.select(conditions, choices, default='Above Average')

        except Exception as e:
            st.warning(f"Error in analytical calculations: {e}")
            df_analytical['price_vs_avg'] = 0
            df_analytical['savings_opportunity'] = 0
            df_analytical['is_cheapest'] = False
            df_analytical['price_ratio'] = 0
            df_analytical['price_segment'] = 'Unknown'

        return df_analytical

    def create_sample_data(self):
        np.random.seed(42)
        products = [
            {'mon': 'MON_00123', 'name': 'Xiaomi Redmi Note 12 128GB', 'base_price': 8500000},
            {'mon': 'MON_00456', 'name': 'Samsung Galaxy A54 5G 256GB', 'base_price': 12000000},
            {'mon': 'MON_00789', 'name': 'Apple iPhone 14 Pro 128GB', 'base_price': 28000000},
            {'mon': 'MON_00234', 'name': 'Huawei P60 Pro 512GB', 'base_price': 18000000},
        ]

        data = []
        for product in products:
            num_sellers = np.random.randint(8, 20)

            for i in range(num_sellers):
                price_variation = np.random.uniform(0.8, 1.3)
                price = int(product['base_price'] * price_variation)

                if product['mon'] == 'MON_00234':
                    platform = 'SnappPay' if i % 2 == 0 else 'Torob'
                else:
                    platform = np.random.choice(list(Config.PLATFORMS.keys()), p=[0.5, 0.5])

                data.append({
                    'MON': product['mon'],
                    'Product Titles': product['name'],
                    'Product Prices': price,
                    'Seller': f'Seller_{i + 1:02d}',
                    'Platform': platform
                })

        df = pd.DataFrame(data)
        df['Product Prices'] = pd.to_numeric(df['Product Prices'], errors='coerce')
        return df


# Step 3: Format Utilities (Same as before)
class FormatUtils:
    @staticmethod
    def format_price(price):
        try:
            return f"{int(float(price)):,} تومان"
        except (ValueError, TypeError):
            return f"{price} تومان"

    @staticmethod
    def format_number(number):
        try:
            return f"{int(float(number)):,}"
        except (ValueError, TypeError):
            return str(number)

    @staticmethod
    def get_product_display_name(mon_value, data_processor):
        if mon_value not in data_processor.product_index:
            return f"{mon_value} (No data)"

        product_info = data_processor.product_index[mon_value]

        platform_icons = ""
        for platform in Config.PLATFORMS.keys():
            if product_info['platform_availability'][platform]:
                platform_icons += Config.PLATFORMS[platform]['icon']

        cross_platform_indicator = " 🔄" if product_info['cross_platform'] else ""

        return f"{mon_value} - {product_info['title']} ({product_info['seller_count']} sellers) {platform_icons}{cross_platform_indicator}"


# Step 4: Enhanced Chart Builder with Platform Statistics (Same as before)
class ChartBuilder:
    @staticmethod
    def create_empty_figure(message="No data available"):
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16),
            xref="paper",
            yref="paper"
        )
        fig.update_layout(
            template=Config.CHART_TEMPLATE,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

    @staticmethod
    def create_main_chart(product_data, selected_product_display, chart_type):
        if product_data.empty:
            return ChartBuilder.create_empty_figure("No data available")

        try:
            plot_data = product_data.copy()
            plot_data = plot_data.dropna(subset=['Product Prices'])

            if plot_data.empty:
                return ChartBuilder.create_empty_figure("No valid price data available")

            if chart_type == 'distribution':
                fig = px.histogram(
                    plot_data,
                    x='Product Prices',
                    color='Platform',
                    title=f'📊 Price Distribution - {selected_product_display}',
                    nbins=15,
                    opacity=0.8,
                    color_discrete_map={platform: Config.PLATFORMS[platform]['color']
                                        for platform in Config.PLATFORMS.keys()}
                )
            elif chart_type == 'seller':
                seller_avg = plot_data.groupby(['Seller', 'Platform'])['Product Prices'].mean().reset_index()
                fig = px.bar(
                    seller_avg,
                    x='Seller',
                    y='Product Prices',
                    color='Platform',
                    title=f'🏪 Seller Prices - {selected_product_display}',
                    color_discrete_map={platform: Config.PLATFORMS[platform]['color']
                                        for platform in Config.PLATFORMS.keys()}
                )
                fig.update_layout(xaxis_tickangle=-45)
            elif chart_type == 'platform':
                fig = px.box(
                    plot_data,
                    x='Platform',
                    y='Product Prices',
                    title=f'🏪 Platform Comparison - {selected_product_display}',
                    color='Platform',
                    color_discrete_map={platform: Config.PLATFORMS[platform]['color']
                                        for platform in Config.PLATFORMS.keys()}
                )
            elif chart_type == 'platform_stats':
                # NEW: Platform Statistics Comparison Chart
                fig = ChartBuilder.create_platform_statistics_chart(plot_data, selected_product_display)
            else:  # Default box plot
                fig = px.box(
                    plot_data,
                    y='Product Prices',
                    color='Platform',
                    title=f'📦 Price Distribution - {selected_product_display}',
                    color_discrete_map={platform: Config.PLATFORMS[platform]['color']
                                        for platform in Config.PLATFORMS.keys()}
                )

            fig.update_layout(template=Config.CHART_TEMPLATE)
            return fig

        except Exception as e:
            st.error(f"Error creating chart: {e}")
            return ChartBuilder.create_empty_figure("Error creating chart")

    @staticmethod
    def create_platform_statistics_chart(product_data, selected_product_display):
        """Create comprehensive platform statistics comparison chart"""
        try:
            # Calculate statistics for each platform
            platform_stats = []

            for platform in product_data['Platform'].unique():
                platform_data = product_data[product_data['Platform'] == platform]
                prices = platform_data['Product Prices']

                stats = {
                    'Platform': platform,
                    'Mean': prices.mean(),
                    'Median': prices.median(),
                    'Min': prices.min(),
                    'Max': prices.max(),
                    'Std': prices.std(),
                    'Variance': prices.var(),
                    'Count': len(prices),
                    'Range': prices.max() - prices.min(),
                    'Q1': prices.quantile(0.25),
                    'Q3': prices.quantile(0.75)
                }
                platform_stats.append(stats)

            stats_df = pd.DataFrame(platform_stats)

            # Create subplots for different statistics
            fig = go.Figure()

            # Define statistics to plot
            statistics_to_plot = ['Mean', 'Min', 'Max', 'Std', 'Variance', 'Range']
            colors = ['#3498db', '#27ae60', '#e74c3c', '#f39c12', '#9b59b6', '#34495e']

            for i, stat in enumerate(statistics_to_plot):
                fig.add_trace(go.Bar(
                    name=stat,
                    x=stats_df['Platform'],
                    y=stats_df[stat],
                    marker_color=colors[i % len(colors)],
                    text=[FormatUtils.format_price(val) if stat in ['Mean', 'Min', 'Max', 'Range']
                          else f"{val:,.0f}" for val in stats_df[stat]],
                    textposition='auto',
                    hovertemplate=f"<b>{stat}</b>: " +
                                  ("%{y:,.0f} تومان" if stat in ['Mean', 'Min', 'Max', 'Range'] else "%{y:,.0f}") +
                                  "<extra></extra>"
                ))

            fig.update_layout(
                title=f'📈 Platform Statistics Comparison - {selected_product_display}',
                xaxis_title='Platform',
                yaxis_title='Value',
                barmode='group',
                template=Config.CHART_TEMPLATE,
                showlegend=True,
                height=500
            )

            return fig

        except Exception as e:
            st.error(f"Error creating platform statistics chart: {e}")
            return ChartBuilder.create_empty_figure("Error creating platform statistics chart")


# Step 5: Component Builder (Adapted for Streamlit)
class ComponentBuilder:
    @staticmethod
    def create_data_table(product_data):
        if product_data.empty:
            st.write("No data available")
            return

        try:
            table_data = product_data[[
                'Seller', 'Platform', 'Product Prices', 'price_segment', 'price_ratio'
            ]].copy().sort_values('Product Prices')

            table_data_formatted = table_data.copy()

            def safe_price_format(x):
                try:
                    return f"{float(x):,.0f} تومان"
                except (ValueError, TypeError):
                    return "N/A"

            def safe_ratio_format(x):
                try:
                    return f"{float(x):+.1f}%"
                except (ValueError, TypeError):
                    return "N/A"

            table_data_formatted['Product Prices'] = table_data_formatted['Product Prices'].apply(safe_price_format)
            table_data_formatted['price_ratio'] = table_data_formatted['price_ratio'].apply(safe_ratio_format)

            # Display as Streamlit dataframe
            st.dataframe(
                table_data_formatted,
                use_container_width=True,
                hide_index=True
            )
        except Exception as e:
            st.error(f"Error creating data table: {e}")

    @staticmethod
    def create_kpi_cards(product_data, selected_product_display, data_processor, mon_value):
        if product_data.empty:
            return

        try:
            total_sellers = product_data['Seller'].nunique()
            prices = pd.to_numeric(product_data['Product Prices'], errors='coerce').dropna()

            if len(prices) == 0:
                st.write("No valid price data available")
                return

            min_price = prices.min()
            max_price = prices.max()
            avg_price = prices.mean()
            price_range = max_price - min_price

            product_info = data_processor.product_index.get(mon_value, {})
            cross_platform = product_info.get('cross_platform', False)
            platform_counts = product_info.get('platform_counts', {})

            # Display main title
            st.subheader(f"📊 {selected_product_display}")

            # Cross-platform indicator
            if cross_platform:
                st.info("🔄 Cross-Platform Product")

            # Create columns for KPI cards
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Total Sellers",
                    value=FormatUtils.format_number(total_sellers)
                )

            with col2:
                st.metric(
                    label="Price Range",
                    value=f"{FormatUtils.format_price(min_price)} - {FormatUtils.format_price(max_price)}"
                )

            with col3:
                st.metric(
                    label="Average Price",
                    value=FormatUtils.format_price(int(avg_price))
                )

            # Platform comparison section
            st.subheader("Platform Comparison")
            platform_cols = st.columns(len(Config.PLATFORMS))

            for i, platform in enumerate(Config.PLATFORMS.keys()):
                platform_data = product_data[product_data['Platform'] == platform]
                platform_prices = pd.to_numeric(platform_data['Product Prices'], errors='coerce').dropna()

                if len(platform_prices) > 0:
                    platform_min = platform_prices.min()
                    platform_sellers = platform_data['Seller'].nunique()

                    with platform_cols[i]:
                        st.write(f"**{Config.PLATFORMS[platform]['icon']} {platform}**")
                        st.write(f"Lowest: {FormatUtils.format_price(platform_min)}")
                        st.write(f"Sellers: {platform_sellers}")

        except Exception as e:
            st.error(f"Error creating KPI cards: {e}")


# Main Streamlit App
def main():
    # Page configuration
    st.set_page_config(
        page_title="Product Price Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">📊 Product Price Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Efficient analysis of product pricing across platforms - Select a product to start</p>',
        unsafe_allow_html=True)

    # Initialize data processor
    if 'data_processor' not in st.session_state:
        with st.spinner('Loading data...'):
            # data_processor = DataProcessor(r"C:\Users\Snapp Pay\Downloads\mon price bf 19k - Merged Files 1 - Copy.csv")
            data_processor = DataProcessor(r"mon.csv")
            st.session_state.data_processor = data_processor

    data_processor = st.session_state.data_processor

    # Search and Filter Section
    col1, col2 = st.columns([2, 1])

    with col1:
        search_terms = st.text_input(
            "🔍 Search Products:",
            placeholder="Search by product name or MON code...",
            help="Enter product name or MON code to filter products"
        )

    with col2:
        sort_order = st.selectbox(
            "Sort Order:",
            options=['ascending', 'descending'],
            format_func=lambda x: '🔼 Sellers: Ascending' if x == 'ascending' else '🔽 Sellers: Descending'
        )

    # Product Selection
    filtered_products = data_processor.search_products(search_terms, sort_order)

    # Display search results count
    if search_terms:
        st.info(f"🔍 Found {FormatUtils.format_number(len(filtered_products))} products matching: '{search_terms}'")
    else:
        st.info(f"📦 Total {FormatUtils.format_number(len(data_processor.product_index))} products available")

    # Product dropdown
    if filtered_products:
        product_options = [
            FormatUtils.get_product_display_name(product, data_processor)
            for product in filtered_products[:200]  # Limit to first 200 for performance
        ]

        selected_display = st.selectbox(
            "Select Product:",
            options=product_options,
            index=0,
            help="Select a product to analyze its pricing data"
        )

        # Extract MON value from selected display
        selected_product = selected_display.split(' - ')[0] if ' - ' in selected_display else selected_display
    else:
        st.warning("No products found matching your search criteria")
        selected_product = None

    # Chart Type Selection
    chart_type = st.selectbox(
        "Chart Type:",
        options=['distribution', 'seller', 'platform', 'box', 'platform_stats'],
        format_func=lambda x: {
            'distribution': '📊 Price Distribution',
            'seller': '🏪 Seller Comparison',
            'platform': '🏪 Platform Comparison',
            'box': '📦 Box Plot',
            'platform_stats': '📈 Platform Statistics'
        }[x],
        index=0
    )

    # Main Content
    if selected_product:
        try:
            # Get product data
            product_data = data_processor.get_product_data(selected_product)

            if not product_data.empty:
                selected_product_display = FormatUtils.get_product_display_name(selected_product, data_processor)

                # Display KPI Cards
                ComponentBuilder.create_kpi_cards(product_data, selected_product_display, data_processor,
                                                  selected_product)

                # Display Chart
                st.subheader("Price Analysis Chart")
                chart = ChartBuilder.create_main_chart(product_data, selected_product_display, chart_type)
                st.plotly_chart(chart, use_container_width=True)

                # Display Data Table
                st.subheader("📋 Detailed Price Data")
                ComponentBuilder.create_data_table(product_data)
            else:
                st.warning("No data available for the selected product")

        except Exception as e:
            st.error(f"Error processing product data: {e}")
    else:
        st.info("Please select a product to view analysis")

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Product Price Analysis Dashboard")


if __name__ == '__main__':
    main()