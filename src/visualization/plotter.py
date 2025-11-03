"""
Unified Visualization Module for Local Energy Market.

This module provides comprehensive visualization capabilities for analyzing and understanding
market patterns, coordination effectiveness, and system performance metrics.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from ..analytics.dso_metrics import DSOMetrics
from ..analytics.grid_metrics import GridMetrics
from ..analytics.market_analytics import MarketMetrics
from ..market.matching import MatchingHistory


class Plotter:
    """Comprehensive market visualization tool."""

    def __init__(self,
                 style: str = "default",
                 save_path: Optional[str] = None,
                 format: str = "png") -> None:
        """Initialize the market visualizer.

        Args:
            style: Matplotlib style to use
            save_path: Optional default path to save plots
            format: Output format for saved plots (png, pdf, svg, jpg, etc.)
        """
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

        self.figsize = (12, 8)
        self.save_path = save_path
        self.format = format

        # Centralized color management
        self.colors = {# Primary colors for price and volume distributions
                       "price_primary": "#4A90E2",
                       "price_secondary": "blue",
                       "volume_primary": "#7ED321",
                       "volume_secondary": "green",

                       # Statistical elements for price and volume distributions
                       "median": "orange",
                       "mean": "red",
                       "percentile": ["orange", "red", "darkred", "purple", "navy"],

                       # Box plot elements for price and volume distributions
                       "box_edge": "black",
                       "whisker": "black",
                       "cap": "black",
                       "flier": "red",

                       # General styling for price and volume distributions
                       "grid": "#e0e0e0",
                       "background": "white",
                       "text": "#333333",
                       "title": "#000000",

                       # Node colors for trading network
                       "buyer": "#4A90E2",
                       "seller": "#7ED321",
                       "balanced": "#FFA500"}

    def _save_figure(self, filename: str, **kwargs) -> None:
        """Helper method to save figures with the configured format.

        Args:
            filename: Base filename without extension
            **kwargs: Additional arguments to pass to plt.savefig
        """
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

            # Remove any existing extension from filename
            base_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
            full_path = f"{self.save_path}/{base_filename}.{self.format}"

            # Set default parameters
            save_kwargs = {"bbox_inches": "tight",
                           "dpi": 300,
                           "facecolor": self.colors["background"]}
            save_kwargs.update(kwargs)

            plt.savefig(full_path, **save_kwargs)

    def trading_network(self,
                        matching_history: MatchingHistory,
                        units: str = "Wh") -> Figure:
        """Create the Local Energy Market Trading Network figure.

        This creates a network graph showing trading relationships between agents with
        three-dimensional visualization:
        - Node color: Net trading position (net buyers=blue, net sellers=green, balanced=orange)
        - Edge thickness: Trading volume between agents
        - Edge color: Distance between agents (light gray=close, dark gray=far)
        - Edge labels: Trading volume values

        Args:
            matching_history: History of market matching results
            units: Units of the trade volumes

        Returns:
            Matplotlib figure
        """
        # Extract all trades from matching history
        all_trades = []
        for result in matching_history.history:
            all_trades.extend(result.trades)

        if not all_trades:
            # Create empty figure if no trades
            fig, ax = plt.subplots(figsize=(12, 8))

            ax.text(0.5,
                    0.5,
                    "No trading data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=16)

            ax.set_title("Local Energy Market Trading Network")

            return fig

        # Create network graph
        G = nx.Graph()

        # Collect all unique agents and calculate their net trading positions
        agent_net_positions = {}  # agent_id -> net_position (positive = net seller, negative = net buyer)
        edge_data = {}

        for trade in all_trades:
            buyer_id = trade.buyer_id
            seller_id = trade.seller_id
            quantity = trade.quantity

            # Calculate net trading positions (when agent A buys from agent B: A becomes more of a buyer (-), B becomes more of a seller (+))
            agent_net_positions[buyer_id] = agent_net_positions.get(buyer_id, 0) - quantity
            agent_net_positions[seller_id] = agent_net_positions.get(seller_id, 0) + quantity

            # Aggregate quantities for edges
            edge_key = (buyer_id, seller_id)
            if edge_key in edge_data:
                edge_data[edge_key] += quantity
            else:
                edge_data[edge_key] = quantity

        # Categorize agents based on net trading position
        net_buyers = []
        net_sellers = []
        balanced_agents = []

        for agent_id, net_position in agent_net_positions.items():
            if net_position > 0:
                # Net seller (sold more than bought)
                net_sellers.append(agent_id)
                G.add_node(agent_id, role="seller", net_position=net_position)

            elif net_position < 0:
                # Net buyer (bought more than sold)
                net_buyers.append(agent_id)
                G.add_node(agent_id, role="buyer", net_position=abs(net_position))

            else:
                # Balanced (equal buying and selling)
                balanced_agents.append(agent_id)
                G.add_node(agent_id, role="balanced", net_position=0)

        for (buyer, seller), quantity in edge_data.items():
            G.add_edge(buyer, seller, weight=quantity)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor("white")

        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

        # Separate nodes by their net trading position
        buyer_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "buyer"]
        seller_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "seller"]
        balanced_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "balanced"]

        # Draw nodes - Net Buyers (blue)
        if buyer_nodes:
            nx.draw_networkx_nodes(G,
                                   pos,
                                   nodelist=buyer_nodes,
                                   node_color=self.colors["buyer"],
                                   node_size=1000,
                                   alpha=0.9,
                                   edgecolors="white",
                                   linewidths=2)

        # Draw nodes - Net Sellers (green)
        if seller_nodes:
            nx.draw_networkx_nodes(G,
                                   pos,
                                   nodelist=seller_nodes,
                                   node_color=self.colors["seller"],
                                   node_size=1000,
                                   alpha=0.9,
                                   edgecolors="white",
                                   linewidths=2)

        # Draw nodes - Balanced Agents (orange)
        if balanced_nodes:
            nx.draw_networkx_nodes(G,
                                   pos,
                                   nodelist=balanced_nodes,
                                   node_color=self.colors["balanced"],
                                   node_size=1000,
                                   alpha=0.9,
                                   edgecolors="white",
                                   linewidths=2)

        # Draw edges with weights and distance-based colors
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        max_weight = max(weights) if weights else 1

        # Normalize edge widths based on trading volume
        edge_widths = [1.5 + 6 * (w / max_weight) for w in weights]

        # Calculate distances and create color mapping using actual trade distances
        edge_colors = []
        distances = []

        # Create a mapping from edge pairs to their distances from trades
        edge_distance_map = {}
        for trade in all_trades:
            edge_key = (trade.buyer_id, trade.seller_id)
            edge_distance_map[edge_key] = trade.distance

        # Get actual distance from trade data
        for u, v in edges:
            distance = edge_distance_map.get((u, v), 0.0)
            distances.append(distance)

        # Normalize distances for color mapping
        if distances and max(distances) > 0:
            min_distance = min(distances)
            max_distance = max(distances)
            distance_range = max_distance - min_distance if max_distance > min_distance else 1

            # Create color mapping using grayscale: close = light gray, far = dark gray
            for distance in distances:
                normalized_distance = (distance - min_distance) / distance_range
                edge_colors.append(plt.cm.Greys(0.3 + 0.7 * normalized_distance))
        else:
            edge_colors = ["silver"] * len(edges)

        nx.draw_networkx_edges(G,
                               pos,
                               width=edge_widths,
                               alpha=0.5,
                               edge_color=edge_colors,
                               style="solid")

        # Add edge labels (quantities)
        edge_labels = {(u, v): f"{G[u][v]["weight"]:.1f} {units}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G,
                                     pos,
                                     edge_labels=edge_labels,
                                     font_size=9,
                                     font_color="black",
                                     bbox=dict(boxstyle="round,pad=0.2",
                                               facecolor="white",
                                               alpha=0.8,
                                               edgecolor="none"))

        # Add node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

        # Create custom legend with proper marker sizes
        legend_handles = []
        if buyer_nodes:
            legend_handles.append(Patch(facecolor=self.colors["buyer"], alpha=0.9, label=f"Buyers"))
        if seller_nodes:
            legend_handles.append(Patch(facecolor=self.colors["seller"], alpha=0.9, label=f"Sellers"))
        if balanced_nodes:
            legend_handles.append(Patch(facecolor=self.colors["balanced"], alpha=0.9, label=f"Balanced"))

        ax.legend(handles=legend_handles,
                  loc="best",
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  fontsize=10,
                  title="Agent Types",
                  title_fontsize=12)

        # Add colorbar for distance scale
        if distances and len(distances) > 1 and max(distances) > 0:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys,
                                       norm=plt.Normalize(vmin=min(distances),
                                                          vmax=max(distances)))
            sm.set_array([])
            cbar = plt.colorbar(sm,
                                ax=ax,
                                shrink=0.6,
                                aspect=20,
                                pad=0.02)

            cbar.set_label('Distance (km)',
                           fontsize=10,
                           rotation=90,
                           labelpad=15)

            cbar.ax.tick_params(labelsize=9)

        plt.title("Local Energy Market Trading Network",
                  fontsize=16,
                  fontweight="bold",
                  pad=20)

        plt.axis("off")

        self._save_figure("trading_network")

        return fig

    def statistical_distribution(self,
                                 matching_history: MatchingHistory,
                                 plot_type: str = "violin") -> Figure:
        """Create the Distribution of Clearing Prices and Trade Volumes figure.

        This creates a side-by-side plot showing the distribution of clearing prices
        and trade volumes from the matching history using consistent visualization methods.

        Args:
            matching_history: History of market matching results
            plot_type: Type of plot to use ("violin", "kde", "histogram", "cdf", "box")

        Returns:
            Matplotlib figure
        """
        # Extract clearing prices and volumes
        clearing_prices = [result.clearing_price for result in matching_history.history]
        trade_volumes = [sum(trade.quantity for trade in result.trades) for result in matching_history.history]

        if not clearing_prices or not trade_volumes:
            # Create empty figure if no data
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            ax1.text(0.5,
                     0.5,
                     "No price data available",
                     ha="center",
                     va="center",
                     transform=ax1.transAxes,
                     fontsize=12)

            ax2.text(0.5,
                     0.5,
                     "No volume data available",
                     ha="center",
                     va="center",
                     transform=ax2.transAxes,
                     fontsize=12)

            ax1.set_title("Distribution of Clearing Prices")
            ax2.set_title("Distribution of Trade Volumes")

            return fig

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor(self.colors["background"])

        # Use centralized colors
        price_color = self.colors["price_primary"]
        volume_color = self.colors["volume_primary"]

        if plot_type == "violin":
            self._violin_distribution(ax1,
                                      ax2,
                                      clearing_prices,
                                      trade_volumes,
                                      price_color,
                                      volume_color)

        elif plot_type == "kde":
            self._kde_distribution(ax1,
                                   ax2,
                                   clearing_prices,
                                   trade_volumes,
                                   price_color,
                                   volume_color)

        elif plot_type == "histogram":
            self._histogram_distribution(ax1,
                                         ax2,
                                         clearing_prices,
                                         trade_volumes,
                                         price_color,
                                         volume_color)

        elif plot_type == "cdf":
            self._cdf_distribution(ax1,
                                   ax2,
                                   clearing_prices,
                                   trade_volumes,
                                   price_color,
                                   volume_color)

        elif plot_type == "box":
            self._box_distribution(ax1,
                                   ax2,
                                   clearing_prices,
                                   trade_volumes,
                                   price_color,
                                   volume_color)

        else:
            raise ValueError(f"Unknown <plot_type = {plot_type}>. Use 'violin', 'kde', 'histogram', 'cdf', or 'box'.")

        # Add overall title
        fig.suptitle("Statistical Distributions of Market Prices and Volumes",
                     fontsize=18,
                     fontweight="bold",
                     y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        self._save_figure(f"statistical_distribution_{plot_type}")

        return fig

    def _violin_distribution(self,
                             ax1: plt.Axes,
                             ax2: plt.Axes,
                             clearing_prices: List[float],
                             trade_volumes: List[float],
                             price_color: str,
                             volume_color: str) -> None:
        """Plot violin distributions for both price and volume.

        Args:
            ax1: Matplotlib axes for price distribution
            ax2: Matplotlib axes for volume distribution
            clearing_prices: List of clearing prices
            trade_volumes: List of trade volumes
            price_color: Color for price distribution
            volume_color: Color for volume distribution
        """
        _alpha = 0.6
        _linewidth = 3

        # Violin plot for prices
        parts1 = ax1.violinplot([clearing_prices],
                                positions=[1],
                                showmeans=True,
                                showmedians=True)

        for pc in parts1["bodies"]:
            pc.set_facecolor(price_color)
            pc.set_alpha(_alpha)
            pc.set_edgecolor(price_color)
            pc.set_linewidth(_linewidth)
            pc.set_zorder(5)

        parts1["cmeans"].set_color(self.colors["mean"])
        parts1["cmeans"].set_linewidth(_linewidth)
        parts1["cmeans"].set_zorder(10)
        parts1["cmedians"].set_color(self.colors["median"])
        parts1["cmedians"].set_linewidth(_linewidth)
        parts1["cmedians"].set_zorder(10)
        parts1["cbars"].set_color("black")
        parts1["cmaxes"].set_color("black")
        parts1["cmins"].set_color("black")

        # Add legend for violin plot elements
        ax1.legend(handles=[Line2D([0], [0], color=self.colors["mean"], linewidth=_linewidth, label=f"Mean: ${np.mean(clearing_prices):.1f}"),
                            Line2D([0], [0], color=self.colors["median"], linewidth=_linewidth, label=f"Median: ${np.median(clearing_prices):.1f}")],
                   loc="upper right",
                   frameon=True,
                   fancybox=True,
                   shadow=False,
                   fontsize=10)

        ax1.set_title("Distribution of Clearing Prices",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"])

        ax1.set_ylabel("Price ($/kWh)")
        ax1.set_xticks([1])
        ax1.set_xticklabels(["Prices"])
        ax1.grid(True, alpha=0.3, zorder=0)

        # Violin plot for volumes
        parts2 = ax2.violinplot([trade_volumes],
                                positions=[1],
                                showmeans=True,
                                showmedians=True)

        for pc in parts2["bodies"]:
            pc.set_facecolor(volume_color)
            pc.set_alpha(_alpha)
            pc.set_edgecolor(volume_color)
            pc.set_linewidth(_linewidth)
            pc.set_zorder(5)

        parts2["cmeans"].set_color(self.colors["mean"])
        parts2["cmeans"].set_linewidth(_linewidth)
        parts2["cmeans"].set_zorder(10)
        parts2["cmedians"].set_color(self.colors["median"])
        parts2["cmedians"].set_linewidth(_linewidth)
        parts2["cmedians"].set_zorder(10)
        parts2["cbars"].set_color("black")
        parts2["cmaxes"].set_color("black")
        parts2["cmins"].set_color("black")

        # Add legend for violin plot elements
        ax2.legend(handles=[Line2D([0], [0], color=self.colors["mean"], linewidth=_linewidth, label=f"Mean: {np.mean(trade_volumes):.1f} kWh"),
                            Line2D([0], [0], color=self.colors["median"], linewidth=_linewidth, label=f"Median: {np.median(trade_volumes):.1f} kWh")],
                   loc="upper right",
                   frameon=True,
                   fancybox=True,
                   shadow=False,
                   fontsize=10)

        ax2.set_title("Distribution of Trade Volumes",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"])
        ax2.set_ylabel("Volume (kWh)")
        ax2.set_xticks([1])
        ax2.set_xticklabels(["Volumes"])
        ax2.grid(True, alpha=0.3, zorder=0)

    def _kde_distribution(self,
                          ax1: plt.Axes,
                          ax2: plt.Axes,
                          clearing_prices: List[float],
                          trade_volumes: List[float],
                          price_color: str,
                          volume_color: str) -> None:
        """Plot KDE distributions for both price and volume.

        Args:
            ax1: Matplotlib axes for price distribution
            ax2: Matplotlib axes for volume distribution
            clearing_prices: List of clearing prices
            trade_volumes: List of trade volumes
            price_color: Color for price distribution
            volume_color: Color for volume distribution
        """
        _alpha = 0.5
        _linewidth = 3

        # KDE for prices
        if len(clearing_prices) > 1:
            kde_prices = gaussian_kde(clearing_prices)
            x_range_prices = np.linspace(min(clearing_prices), max(clearing_prices), 200)
            density_prices = kde_prices(x_range_prices)

            ax1.fill_between(x_range_prices,
                             density_prices,
                             alpha=_alpha,
                             color=price_color,
                             zorder=5)

            ax1.plot(x_range_prices,
                     density_prices,
                     color=price_color,
                     linewidth=_linewidth,
                     zorder=5)

            # Add statistics
            mean_price = np.mean(clearing_prices)
            median_price = np.median(clearing_prices)

            ax1.axvline(mean_price,
                        color=self.colors["mean"],
                        linestyle="-",
                        linewidth=_linewidth,
                        label=f"Mean: ${mean_price:.1f}",
                        zorder=5)

            ax1.axvline(median_price,
                        color=self.colors["median"],
                        linestyle="-",
                        linewidth=_linewidth,
                        label=f"Median: ${median_price:.1f}",
                        zorder=5)

            ax1.legend()

        ax1.set_title("Distribution of Clearing Prices", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Price ($/kWh)")
        ax1.set_ylabel("Density")
        ax1.set_ylim(bottom=0)
        ax1.grid(True, alpha=0.3, zorder=0)

        # KDE for volumes
        if len(trade_volumes) > 1:
            kde_volumes = gaussian_kde(trade_volumes)
            x_range_volumes = np.linspace(min(trade_volumes), max(trade_volumes), 200)
            density_volumes = kde_volumes(x_range_volumes)

            ax2.fill_between(x_range_volumes,
                             density_volumes,
                             alpha=_alpha,
                             color=volume_color,
                             zorder=5)

            ax2.plot(x_range_volumes,
                     density_volumes,
                     color=volume_color,
                     linewidth=_linewidth,
                     zorder=5)

            # Add statistics
            mean_volume = np.mean(trade_volumes)
            median_volume = np.median(trade_volumes)

            ax2.axvline(mean_volume,
                        color=self.colors["mean"],
                        linestyle="-",
                        linewidth=_linewidth,
                        label=f"Mean: {mean_volume:.1f} kWh",
                        zorder=5)

            ax2.axvline(median_volume,
                        color=self.colors["median"],
                        linestyle="-",
                        linewidth=_linewidth,
                        label=f"Median: {median_volume:.1f} kWh",
                        zorder=5)

            ax2.legend()

        ax2.set_title("Distribution of Trade Volumes", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Volume (kWh)")
        ax2.set_ylabel("Density")
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3, zorder=0)

    def _histogram_distribution(self,
                                ax1: plt.Axes,
                                ax2: plt.Axes,
                                clearing_prices: List[float],
                                trade_volumes: List[float]) -> None:
        """Plot histogram + KDE distributions for both price and volume (same as box style).

        Args:
            ax1: Matplotlib axes for price distribution
            ax2: Matplotlib axes for volume distribution
            clearing_prices: List of clearing prices
            trade_volumes: List of trade volumes
        """
        _alpha = 0.6
        _linewidth = 3

        # Create histogram with frequency (left y-axis)
        counts, bins, patches = ax1.hist(clearing_prices,
                                         bins=min(20, len(clearing_prices) // 2) if len(clearing_prices) > 4 else 10,
                                         alpha=_alpha,
                                         color=self.colors["price_primary"],
                                         edgecolor=self.colors["box_edge"],
                                         linewidth=1,
                                         density=False,  # Frequency
                                         zorder=5)

        # Create secondary y-axis for density
        ax1_twin = ax1.twinx()

        # Add KDE overlay on density axis
        if len(clearing_prices) > 1:
            kde = gaussian_kde(clearing_prices)
            x_range = np.linspace(min(clearing_prices), max(clearing_prices), 200)

            ax1_twin.plot(x_range,
                          kde(x_range),
                          color=self.colors["price_secondary"],
                          linewidth=_linewidth,
                          label="KDE",
                          zorder=5)

        ax1.set_title("Distribution of Clearing Prices",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"])

        ax1.set_xlabel("Price ($/kWh)", color=self.colors["text"])
        ax1.set_ylabel("Frequency", color=self.colors["text"])
        ax1_twin.set_ylabel("Density", color=self.colors["text"])
        ax1.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

        # Create histogram with frequency (left y-axis)
        counts, bins, patches = ax2.hist(trade_volumes,
                                         bins=min(20, len(trade_volumes) // 2) if len(trade_volumes) > 4 else 10,
                                         alpha=_alpha,
                                         color=self.colors["volume_primary"],
                                         edgecolor=self.colors["box_edge"],
                                         linewidth=1,
                                         density=False,
                                         zorder=5)

        # Create secondary y-axis for density
        ax2_twin = ax2.twinx()

        # Add KDE overlay on density axis
        if len(trade_volumes) > 1:
            kde = gaussian_kde(trade_volumes)
            x_range = np.linspace(min(trade_volumes), max(trade_volumes), 200)

            ax2_twin.plot(x_range,
                          kde(x_range),
                          color=self.colors["volume_secondary"],
                          linewidth=_linewidth,
                          label="KDE",
                          zorder=5)

        ax2.set_title("Distribution of Trade Volumes",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"])

        ax2.set_xlabel("Volume (kWh)", color=self.colors["text"])
        ax2.set_ylabel("Frequency", color=self.colors["text"])
        ax2_twin.set_ylabel("Density", color=self.colors["text"])
        ax2.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

    def _cdf_distribution(self,
                          ax1: plt.Axes,
                          ax2: plt.Axes,
                          clearing_prices: List[float],
                          trade_volumes: List[float],
                          price_color: str,
                          volume_color: str) -> None:
        """Plot CDF distributions for both price and volume.

        Args:
            ax1: Matplotlib axes for price distribution
            ax2: Matplotlib axes for volume distribution
            clearing_prices: List of clearing prices
            trade_volumes: List of trade volumes
            price_color: Color for price distribution
            volume_color: Color for volume distribution
        """
        # CDF for prices
        sorted_prices = np.sort(clearing_prices)
        y_prices = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)
        ax1.plot(sorted_prices,
                 y_prices,
                 color=price_color,
                 linewidth=3,
                 marker="o",
                 markersize=6,
                 markerfacecolor="white",
                 markeredgecolor=price_color,
                 markeredgewidth=2.5,
                 label="CDF",
                 zorder=5)

        # Add percentiles with better styling
        percentiles = [25, 50, 75, 90, 95]
        percentile_colors = self.colors["percentile"]
        legend_elements = [Line2D([0], [0], color=price_color, linewidth=3, label="CDF")]

        for i, p in enumerate(percentiles):
            value = np.percentile(clearing_prices, p)
            color = percentile_colors[i]

            ax1.axvline(value,
                        color=color,
                        linestyle="-",
                        alpha=0.8,
                        linewidth=1.2)

            ax1.text(value,
                     p/100 - 0.06,
                     f"{p}%",
                     rotation=90,
                     va="bottom",
                     ha="right",
                     fontsize=8,
                     fontweight="bold",
                     color=color)

            legend_elements.append(Line2D([0], [0], color=color, linestyle="--", linewidth=3, label=f"{p}th percentile"))

        ax1.set_title("Cumulative Distribution of Clearing Prices",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"])

        ax1.set_xlabel("Price ($/kWh)")
        ax1.set_ylabel("Cumulative Probability")
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.set_ylim(0, 1)

        # CDF for volumes
        sorted_volumes = np.sort(trade_volumes)
        y_volumes = np.arange(1, len(sorted_volumes) + 1) / len(sorted_volumes)

        ax2.plot(sorted_volumes,
                 y_volumes,
                 color=volume_color,
                 linewidth=3,
                 marker="o",
                 markersize=6,
                 markerfacecolor="white",
                 markeredgecolor=volume_color,
                 markeredgewidth=2.5,
                 alpha=0.9,
                 label="CDF")

        # Add percentiles with better styling
        legend_elements = [Line2D([0], [0], color=volume_color, linewidth=3, label="CDF")]

        for i, p in enumerate(percentiles):
            value = np.percentile(trade_volumes, p)
            color = self.colors["percentile"][i]

            ax2.axvline(value,
                        color=color,
                        linestyle="-",
                        alpha=0.7,
                        linewidth=1.5)

            ax2.text(value,
                     p/100 - 0.06,
                     f"{p}%",
                     rotation=90,
                     va="bottom",
                     ha="right",
                     fontsize=8,
                     fontweight="bold",
                     color=color)

            legend_elements.append(Line2D([0], [0], color=color, linestyle="-", linewidth=3, label=f"{p}th percentile"))

        ax2.set_title("Cumulative Distribution of Trade Volumes",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"])

        ax2.set_xlabel("Volume (kWh)")
        ax2.set_ylabel("Cumulative Probability")
        ax2.grid(True, alpha=0.3, zorder=0)
        ax2.set_ylim(0, 1)

    def _box_distribution(self,
                          ax1: plt.Axes,
                          ax2: plt.Axes,
                          clearing_prices: List[float],
                          trade_volumes: List[float]) -> None:
        """Plot box plots with whiskers for both price and volume.

        Args:
            ax1: Matplotlib axes for price distribution
            ax2: Matplotlib axes for volume distribution
            clearing_prices: List of clearing prices
            trade_volumes: List of trade volumes
        """
        # Box plot for prices - matching the original style
        ax1.boxplot([clearing_prices],
                    patch_artist=True,
                    boxprops={"facecolor": self.colors["price_primary"],
                              "alpha": 0.6,
                              "linewidth": 1,
                              "edgecolor": self.colors["box_edge"]},
                    medianprops={"color": self.colors["median"],
                                 "linewidth": 3},
                    whiskerprops={"linewidth": 1,
                                  "color": self.colors["whisker"]},
                    capprops={"linewidth": 1,
                              "color": self.colors["cap"]},
                    flierprops={"marker": "o",
                                "markerfacecolor": self.colors["flier"],
                                "markersize": 4,
                                "alpha": 0.7,
                                "markeredgecolor": self.colors["box_edge"]})

        # Add legend for price plot (median only)
        legend_elements = [Line2D([0], [0], color=self.colors["median"], linestyle="-", linewidth=2, label="Median")]

        ax1.legend(handles=legend_elements,
                   loc="upper right",
                   frameon=True,
                   fancybox=True,
                   shadow=False,
                   fontsize=10)

        ax1.set_title("Distribution of Clearing Prices",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"])

        ax1.set_ylabel("Price ($/kWh)", color=self.colors["text"])
        ax1.set_xticks([1])
        ax1.set_xticklabels(["Prices"])
        ax1.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

        # Box plot for volumes - matching the original style
        ax2.boxplot([trade_volumes],
                    patch_artist=True,
                    boxprops={"facecolor": self.colors["volume_primary"],
                              "alpha": 0.6,
                              "linewidth": 1,
                              "edgecolor": self.colors["box_edge"]},
                    medianprops={"color": self.colors["median"],
                                 "linewidth": 3},
                    whiskerprops={"color": self.colors["whisker"],
                                  "linewidth": 1},
                    capprops={"color": self.colors["cap"],
                              "linewidth": 1},
                    flierprops={"marker": "o",
                                "markerfacecolor": self.colors["flier"],
                                "markersize": 4,
                                "alpha": 0.7,
                                "markeredgecolor": self.colors["box_edge"]})

        # Add legend for volume plot (median only)
        ax2.legend(handles=legend_elements,
                   loc="upper right",
                   frameon=True,
                   fancybox=True,
                   shadow=False,
                   fontsize=10)

        ax2.set_title("Distribution of Trade Volumes",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"])
        ax2.set_ylabel("Volume (kWh)", color=self.colors["text"])
        ax2.set_xticks([1])
        ax2.set_xticklabels(["Volumes"])
        ax2.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

    def spatial_heatmap(self,
                        matching_history: MatchingHistory,
                        grid_size: int = None) -> Figure:
        """Create comprehensive Agent-to-Agent Trading Analysis with four heatmaps.

        This creates four separate heatmaps showing:
        1. Trading volume activity between agents
        2. Price × quantity trading activity between agents
        3. Trade count activity between agents
        4. Distance matrix between agents (if grid network available)

        Args:
            matching_history: History of market matching results
            grid_size: Size of the spatial grid. If None, will be set to number of unique agents

        Returns:
            Matplotlib figure (returns the first figure - volume activity)
        """
        # Extract all trades from all market steps
        all_trades = []
        for result in matching_history.history:
            all_trades.extend(result.trades)

        if not all_trades:
            # Create empty heatmap if no trades
            fig, ax = plt.subplots(figsize=(10, 8))

            sns.heatmap(np.zeros((10, 10)),
                        cmap="YlOrRd",
                        cbar_kws={"label": "Trading Volume (kWh)"})

            ax.set_title("Agent-to-Agent Trading Matrix", fontsize=16, fontweight="bold")
            ax.set_xlabel("Sellers")
            ax.set_ylabel("Buyers")
            ax.grid(False)

            return fig

        # Get all unique agents and create ordered list
        unique_agents = set()
        for trade in all_trades:
            unique_agents.add(trade.buyer_id)
            unique_agents.add(trade.seller_id)

        agent_list = sorted(list(unique_agents))
        num_agents = len(agent_list)

        # Set grid size to number of unique agents if not specified
        if grid_size is None:
            grid_size = num_agents

        # Create trading matrices: [buyer_idx, seller_idx] = accumulated_value
        volume_matrix = np.zeros((num_agents, num_agents))
        price_matrix = np.zeros((num_agents, num_agents))
        count_matrix = np.zeros((num_agents, num_agents))

        # Create net trading position matrices to show buyer vs seller behavior
        net_position_matrix = np.zeros((num_agents, num_agents))
        net_price_matrix = np.zeros((num_agents, num_agents))

        # Populate the matrices with accumulated data across all steps
        for trade in all_trades:
            buyer_idx = agent_list.index(trade.buyer_id)
            seller_idx = agent_list.index(trade.seller_id)

            # Accumulate trading volume: buyer_idx (row) buys from seller_idx (column)
            volume_matrix[buyer_idx, seller_idx] += trade.quantity

            # Accumulate price × quantity
            price_matrix[buyer_idx, seller_idx] += trade.quantity * trade.price

            # Accumulate trade count
            count_matrix[buyer_idx, seller_idx] += 1

            # Net position: positive = net seller, negative = net buyer
            # When agent A buys from agent B: A becomes more of a buyer (-), B becomes more of a seller (+)
            net_position_matrix[buyer_idx, seller_idx] -= trade.quantity  # Buyer gets negative value
            net_position_matrix[seller_idx, buyer_idx] += trade.quantity  # Seller gets positive value

            # Net price × quantity position: same logic but with price × quantity values
            net_price_matrix[buyer_idx, seller_idx] -= trade.quantity * trade.price  # Buyer gets negative value
            net_price_matrix[seller_idx, buyer_idx] += trade.quantity * trade.price  # Seller gets positive value

        # 1. Volume Activity Figure
        fig1 = self._create_heatmap(volume_matrix,
                                    "Agent-to-Agent Volume Trading Matrix",
                                    "Trading Volume (kWh)",
                                    "agent_volume_matrix.png",
                                    num_agents=num_agents,
                                    agent_list=agent_list)

        # 2. Price × Quantity Activity Figure
        fig2 = self._create_heatmap(price_matrix,
                                    "Agent-to-Agent Price × Quantity Matrix",
                                    "Price × Quantity ($)",
                                    "agent_price_matrix.png",
                                    num_agents=num_agents,
                                    agent_list=agent_list)

        # 3. Trade Count Activity Figure
        fig3 = self._create_heatmap(count_matrix,
                                    "Agent-to-Agent Trade Count Matrix",
                                    "Number of Trades",
                                    "agent_count_matrix.png",
                                    num_agents=num_agents,
                                    agent_list=agent_list)

        # Return the figures
        return fig1, fig2, fig3

    def _create_heatmap(self,
                        matrix: np.ndarray,
                        title: str,
                        cbar_label: str,
                        filename: str,
                        use_coolwarm: bool = False,
                        num_agents: int = None,
                        agent_list: List[str] = None) -> Figure:
        """Create a heatmap visualization.

        Args:
            matrix: 2D numpy array of values
            title: Plot title
            cbar_label: Label for color bar
            filename: Filename to save the plot
            use_coolwarm: Whether to use coolwarm colormap
            num_agents: Number of agents (for tick positioning)
            agent_list: List of agent IDs (for tick labels)
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(self.colors["background"])

        sns.heatmap(matrix,
                    cmap="coolwarm" if use_coolwarm else "YlOrRd",
                    center=0 if use_coolwarm else None,
                    cbar_kws={"label": cbar_label, "shrink": 0.8},
                    ax=ax,
                    square=True,
                    linewidths=0.5,
                    linecolor="white",
                    alpha=0.9)

        ax.set_title(title,
                     fontsize=18,
                     fontweight="bold",
                     color=self.colors["title"],
                     pad=20)
        ax.set_xlabel("Sellers",
                      fontsize=14,
                      color=self.colors["text"])
        ax.set_ylabel("Buyers",
                      fontsize=14,
                      color=self.colors["text"])

        # Set tick labels - show agent IDs centered in each slot
        if num_agents is not None and agent_list is not None:
            tick_positions = [i + 0.5 for i in range(num_agents)]  # Center of each slot
            tick_labels = agent_list

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=10)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, rotation=45, fontsize=10)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, linewidth=0.5)

        self._save_figure(filename)

        return fig

    def plot_grid_stability_metrics(self, metrics: GridMetrics) -> Figure:
        """Create a comprehensive plot of grid stability metrics with time series analysis.

        Args:
            metrics: GridMetrics object

        Returns:
            Matplotlib figure
        """
        # Check if we have time series data
        has_time_series = (metrics.grid_balance_over_time and
                           metrics.grid_congestion_over_time and
                           metrics.grid_stability_over_time and
                           metrics.grid_utilization_over_time)

        if has_time_series:
            # Enhanced layout with time series plots
            fig = plt.figure(figsize=(20, 16))
            fig.patch.set_facecolor(self.colors["background"])
            gs = plt.GridSpec(3, 2, hspace=0.3, wspace=0.3, height_ratios=[1, 1, 1])

            # Define color palette
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

            # Time series data
            time_steps = list(range(len(metrics.grid_balance_over_time)))

            # 1. Grid Balance Over Time (Top-Left)
            ax1 = fig.add_subplot(gs[0, 0])

            ax1.plot(time_steps,
                     metrics.grid_balance_over_time,
                     color=colors[0],
                     linewidth=2.5,
                     marker='o',
                     markersize=4,
                     markerfacecolor='white',
                     markeredgecolor=colors[0],
                     markeredgewidth=1.5)

            ax1.axhline(y=0,
                        color='red',
                        linestyle='--',
                        alpha=0.7,
                        linewidth=2,
                        label='Perfect Balance')

            ax1.set_title("Grid Balance Over Time", fontsize=14, fontweight="bold", color=self.colors["title"])
            ax1.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
            ax1.set_ylabel("Grid Balance", fontsize=12, color=self.colors["text"])
            ax1.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)
            ax1.legend(loc="best", fontsize=10)

            # 2. Grid Congestion Over Time (Top-Right)
            ax2 = fig.add_subplot(gs[0, 1])

            ax2.plot(time_steps,
                     metrics.grid_congestion_over_time,
                     color=colors[1],
                     linewidth=2.5,
                     marker='s',
                     markersize=4,
                     markerfacecolor='white',
                     markeredgecolor=colors[1],
                     markeredgewidth=1.5)

            ax2.set_title("Grid Congestion Over Time",
                          fontsize=14,
                          fontweight="bold",
                          color=self.colors["title"])

            ax2.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
            ax2.set_ylabel("Congestion Level", fontsize=12, color=self.colors["text"])
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

            # 3. Grid Stability Index Over Time (Bottom-Left)
            ax3 = fig.add_subplot(gs[1, 0])

            ax3.plot(time_steps,
                     metrics.grid_stability_over_time,
                     color=colors[2],
                     linewidth=2.5,
                     marker='^',
                     markersize=4,
                     markerfacecolor='white',
                     markeredgecolor=colors[2],
                     markeredgewidth=1.5)

            ax3.set_title("Grid Stability Index",
                          fontsize=14,
                          fontweight="bold",
                          color=self.colors["title"])

            ax3.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
            ax3.set_ylabel("Stability Score", fontsize=12, color=self.colors["text"])
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

            # 4. Grid Utilization Over Time (Bottom-Right)
            ax4 = fig.add_subplot(gs[1, 1])

            ax4.plot(time_steps, metrics.grid_utilization_over_time,
                     color=colors[3],
                     linewidth=2.5,
                     marker='D',
                     markersize=4,
                     markerfacecolor='white',
                     markeredgecolor=colors[3],
                     markeredgewidth=1.5)

            ax4.set_title("Grid Utilization",
                          fontsize=14,
                          fontweight="bold",
                          color=self.colors["title"])
            ax4.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
            ax4.set_ylabel("Utilization Ratio", fontsize=12, color=self.colors["text"])

            ax4.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

            # 5. Summary Metrics (Bottom Row, Full Width)
            ax5 = fig.add_subplot(gs[2, :])
            summary_metrics = {"Avg Congestion": metrics.avg_congestion_level,
                               "Transmission Loss Ratio": metrics.transmission_loss_ratio,
                               "Grid Utilization": metrics.grid_utilization_efficiency,
                               "Load Factor": metrics.load_factor,
                               "Capacity Utilization": metrics.capacity_utilization}

            y_pos = np.arange(len(summary_metrics))
            values = list(summary_metrics.values())
            bar_colors = [colors[i % len(colors)] for i in range(len(summary_metrics))]

            bars = ax5.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(summary_metrics.keys(), fontsize=11)
            ax5.set_xlabel("Value", fontsize=12, color=self.colors["text"])
            ax5.set_title("Grid Performance Summary Metrics", fontsize=14, fontweight="bold", color=self.colors["title"])
            ax5.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0, axis='x')

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax5.text(value + 0.01,
                         bar.get_y() + bar.get_height()/2,
                         f'{value:.3f}',
                         va='center',
                         ha='left',
                         fontweight='bold',
                         fontsize=10)

            # Clean up spines and styling for all axes
            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color(self.colors["text"])
                ax.spines['bottom'].set_color(self.colors["text"])
                ax.tick_params(colors=self.colors["text"])

        else:
            # Fallback to original layout when no time series data
            fig = plt.figure(figsize=(16, 12))
            fig.patch.set_facecolor(self.colors["background"])
            gs = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

            # Define color palette inspired by reference figures
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

            # Grid stability index
            ax1 = fig.add_subplot(gs[0, 0])
            stability_color = "#2ca02c" if metrics.grid_stability_index > 0.8 else "#ff7f0e" if metrics.grid_stability_index > 0.6 else "#d62728"

            ax1.bar(["Grid Stability"],
                    [metrics.grid_stability_index],
                    color=stability_color,
                    alpha=0.8,
                    edgecolor='white',
                    linewidth=2)

            ax1.set_title("Grid Stability Index", fontsize=14, fontweight="bold", color=self.colors["title"])
            ax1.set_ylabel("Index Value", fontsize=12, color=self.colors["text"])
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

            # Add value on top of bar
            ax1.text(0,
                     metrics.grid_stability_index + 0.02,
                     f'{metrics.grid_stability_index:.3f}',
                     ha='center',
                     va='bottom',
                     fontweight='bold',
                     fontsize=11)

            # Supply-demand balance
            ax2 = fig.add_subplot(gs[0, 1])
            balance_color = "#2ca02c" if abs(metrics.supply_demand_balance) < 0.1 else "#ff7f0e" if abs(metrics.supply_demand_balance) < 0.3 else "#d62728"

            ax2.bar(["Supply-Demand Balance"],
                    [metrics.supply_demand_balance],
                    color=balance_color,
                    alpha=0.8,
                    edgecolor='white',
                    linewidth=2)

            ax2.set_title("Supply-Demand Balance", fontsize=14, fontweight="bold", color=self.colors["title"])
            ax2.set_ylabel("Balance", fontsize=12, color=self.colors["text"])
            ax2.set_ylim(-1, 1)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, zorder=0)
            ax2.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

            # Add value on top of bar
            ax2.text(0,
                     metrics.supply_demand_balance + (0.05 if metrics.supply_demand_balance >= 0 else -0.05),
                    f'{metrics.supply_demand_balance:.3f}',
                    ha='center',
                    va='bottom' if metrics.supply_demand_balance >= 0 else 'top',
                    fontweight='bold',
                    fontsize=11)

            # Performance metrics
            ax3 = fig.add_subplot(gs[1, :])
            performance_metrics = {"Avg Congestion": metrics.avg_congestion_level,
                                   "Transmission Loss Ratio": metrics.transmission_loss_ratio,
                                   "Grid Utilization": metrics.grid_utilization_efficiency,
                                   "Load Factor": metrics.load_factor,
                                   "Capacity Utilization": metrics.capacity_utilization}

            # Create horizontal bar plot with better colors
            y_pos = np.arange(len(performance_metrics))
            values = list(performance_metrics.values())
            bar_colors = [colors[i % len(colors)] for i in range(len(performance_metrics))]

            bars3 = ax3.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(performance_metrics.keys(), fontsize=11)
            ax3.set_xlabel("Value", fontsize=12, color=self.colors["text"])
            ax3.set_title("Grid Performance Metrics", fontsize=14, fontweight="bold", color=self.colors["title"])
            ax3.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0, axis='x')

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars3, values)):
                ax3.text(value + 0.01,
                         bar.get_y() + bar.get_height()/2,
                         f'{value:.3f}',
                         va='center',
                         ha='left',
                         fontweight='bold',
                         fontsize=10)

            # Clean up spines and styling
            for ax in [ax1, ax2, ax3]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color(self.colors["text"])
                ax.spines['bottom'].set_color(self.colors["text"])
                ax.tick_params(colors=self.colors["text"])

        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        self._save_figure("grid_stability_metrics")
        return fig

    def plot_economic_efficiency_metrics(self, metrics: MarketMetrics) -> Figure:
        """Create a comprehensive plot of economic efficiency metrics.

        Args:
            metrics: MarketMetrics object

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor(self.colors["background"])
        gs = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        # Define color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Price metrics
        ax1 = fig.add_subplot(gs[0, 0])

        bars1 = ax1.bar(["Average Clearing Price"],
                        [metrics.avg_clearing_price],
                        color=colors[0],
                        alpha=0.8,
                        edgecolor='white',
                        linewidth=2)

        ax1.set_title("Market Clearing Price", fontsize=14, fontweight="bold", color=self.colors["title"])
        ax1.set_ylabel("Price ($/kWh)", fontsize=12, color=self.colors["text"])
        ax1.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

        # Add value on top of bar
        ax1.text(0, metrics.avg_clearing_price + max(metrics.avg_clearing_price * 0.05, 1),
                f'${metrics.avg_clearing_price:.2f}',
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=11)

        # Market quality metrics
        ax2 = fig.add_subplot(gs[0, 1])
        market_metrics = {"Price Volatility": metrics.price_volatility,
                          "Market Liquidity": metrics.market_liquidity,
                          "Market Concentration": metrics.market_concentration}

        y_pos = np.arange(len(market_metrics))
        values = list(market_metrics.values())
        bar_colors = [colors[i % len(colors)] for i in range(len(market_metrics))]

        bars2 = ax2.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(market_metrics.keys(), fontsize=11)
        ax2.set_xlabel("Value", fontsize=12, color=self.colors["text"])
        ax2.set_title("Market Quality Metrics", fontsize=14, fontweight="bold", color=self.colors["title"])
        ax2.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0, axis='x')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars2, values)):
            ax2.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                    va='center',
                    ha='left',
                    fontweight='bold',
                    fontsize=10)

        # Surplus distribution
        ax3 = fig.add_subplot(gs[1, 0])
        surplus_data = {"Consumer Surplus": metrics.consumer_surplus,
                        "Producer Surplus": metrics.producer_surplus}

        if sum(surplus_data.values()) > 0:
            wedges, texts, autotexts = ax3.pie(surplus_data.values(),
                                             labels=surplus_data.keys(),
                                             autopct="%1.1f%%",
                                             colors=[colors[2], colors[3]],
                                             startangle=90,
                                             textprops={'fontsize': 11})

            # Improve text styling
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
        else:
            ax3.text(0.5,
                     0.5,
                     "No surplus data available",
                     ha="center",
                     va="center",
                     transform=ax3.transAxes,
                     fontsize=12,
                     color=self.colors["text"])
        ax3.set_title("Surplus Distribution", fontsize=14, fontweight="bold", color=self.colors["title"])

        # Trading metrics
        ax4 = fig.add_subplot(gs[1, 1])
        trading_metrics = {"Total Volume": metrics.total_trading_volume,
                           "Transaction Count": metrics.transaction_count,
                           "Avg Trade Size": metrics.avg_trade_size}

        y_pos = np.arange(len(trading_metrics))
        values = list(trading_metrics.values())
        bar_colors = [colors[i % len(colors)] for i in range(len(trading_metrics))]

        bars4 = ax4.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(trading_metrics.keys(), fontsize=11)
        ax4.set_xlabel("Value", fontsize=12, color=self.colors["text"])
        ax4.set_title("Trading Metrics", fontsize=14, fontweight="bold", color=self.colors["title"])
        ax4.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0, axis='x')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars4, values)):
            ax4.text(value + max(value * 0.05, 1),
                     bar.get_y() + bar.get_height()/2,
                     f'{value:.1f}',
                     va='center',
                     ha='left',
                     fontweight='bold',
                     fontsize=10)

        # Clean up spines and styling
        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.colors["text"])
            ax.spines['bottom'].set_color(self.colors["text"])
            ax.tick_params(colors=self.colors["text"])

        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        self._save_figure("economic_efficiency_metrics")

        return fig

    def plot_resource_utilization_metrics(self, metrics: MarketMetrics) -> Figure:
        """Create a comprehensive plot of resource utilization metrics.

        Args:
            metrics: MarketMetrics object

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor(self.colors["background"])
        gs = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        # Define color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Market efficiency metrics
        ax1 = fig.add_subplot(gs[0, 0])
        efficiency_metrics = {"Allocation Efficiency": metrics.allocation_efficiency,
                              "Price Discovery Efficiency": metrics.price_discovery_efficiency}

        y_pos = np.arange(len(efficiency_metrics))
        values = list(efficiency_metrics.values())
        bar_colors = [colors[i % len(colors)] for i in range(len(efficiency_metrics))]

        bars1 = ax1.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(efficiency_metrics.keys(), fontsize=11)
        ax1.set_xlabel("Efficiency Score", fontsize=12, color=self.colors["text"])
        ax1.set_title("Market Efficiency", fontsize=14, fontweight="bold", color=self.colors["title"])
        ax1.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0, axis='x')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, values)):
            ax1.text(value + 0.01,
                     bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                     va='center',
                     ha='left',
                     fontweight='bold',
                     fontsize=10)

        # Welfare metrics
        ax2 = fig.add_subplot(gs[0, 1])
        welfare_metrics = {"Social Welfare": metrics.social_welfare / 1000,  # Normalize for display
                           "Welfare Gini": metrics.welfare_distribution_gini,
                           "Price Fairness": metrics.price_fairness_index}

        y_pos = np.arange(len(welfare_metrics))
        values = list(welfare_metrics.values())
        bar_colors = [colors[i % len(colors)] for i in range(len(welfare_metrics))]

        bars2 = ax2.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(welfare_metrics.keys(), fontsize=11)
        ax2.set_xlabel("Value", fontsize=12, color=self.colors["text"])
        ax2.set_title("Welfare Metrics", fontsize=14, fontweight="bold", color=self.colors["title"])
        ax2.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0, axis='x')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars2, values)):
            ax2.text(value + max(value * 0.05, 0.01),
                     bar.get_y() + bar.get_height()/2,
                     f'{value:.3f}',
                     va='center',
                     ha='left',
                     fontweight='bold',
                     fontsize=10)

        # Market structure
        ax3 = fig.add_subplot(gs[1, 0])
        structure_metrics = {"P2P Trade Ratio": metrics.p2p_trade_ratio,
                             "DSO Dependency": metrics.dso_dependency_ratio,
                             "Market Liquidity": metrics.market_liquidity}

        y_pos = np.arange(len(structure_metrics))
        values = list(structure_metrics.values())

        # Use conditional colors based on performance
        bar_colors = []
        for value in values:
            if value > 0.7:
                bar_colors.append(colors[2])  # Green for good performance
            elif value > 0.3:
                bar_colors.append(colors[1])  # Orange for medium performance
            else:
                bar_colors.append(colors[3])  # Red for poor performance

        bars3 = ax3.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(structure_metrics.keys(), fontsize=11)
        ax3.set_xlabel("Ratio/Index", fontsize=12, color=self.colors["text"])
        ax3.set_title("Market Structure", fontsize=14, fontweight="bold", color=self.colors["title"])
        ax3.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0, axis='x')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars3, values)):
            ax3.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{value:.3f}',
                     va='center',
                     ha='left',
                     fontweight='bold',
                     fontsize=10)

        # Trading activity
        ax4 = fig.add_subplot(gs[1, 1])
        trading_metrics = {"Total Volume": metrics.total_trading_volume / 1000,  # Normalize for display
                           "Transaction Count": metrics.transaction_count,
                           "Avg Trade Size": metrics.avg_trade_size}

        y_pos = np.arange(len(trading_metrics))
        values = list(trading_metrics.values())
        bar_colors = [colors[i % len(colors)] for i in range(len(trading_metrics))]

        bars4 = ax4.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(trading_metrics.keys(), fontsize=11)
        ax4.set_xlabel("Value", fontsize=12, color=self.colors["text"])
        ax4.set_title("Trading Activity", fontsize=14, fontweight="bold", color=self.colors["title"])
        ax4.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0, axis='x')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars4, values)):
            ax4.text(value + max(value * 0.05, 1),
                     bar.get_y() + bar.get_height()/2,
                     f'{value:.1f}',
                     va='center',
                     ha='left',
                     fontweight='bold',
                     fontsize=10)

        # Clean up spines and styling
        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.colors["text"])
            ax.spines['bottom'].set_color(self.colors["text"])
            ax.tick_params(colors=self.colors["text"])

        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        self._save_figure("resource_utilization_metrics")

        return fig

    def plot_dso_metrics(self, metrics: DSOMetrics) -> Figure:
        """Create a comprehensive plot of DSO fallback mechanism performance metrics.

        Args:
            metrics: DSOMetrics object

        Returns:
            Matplotlib figure
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1.2])

        # 1. Trade distribution pie chart (local vs DSO)
        ax1 = fig.add_subplot(gs[0, 0])
        trade_distribution = {"Local Market": metrics.p2p_trade_ratio * 100,
                              "DSO Fallback": metrics.dso_trade_ratio * 100}

        # Only create pie chart if there"s actual data
        if sum(trade_distribution.values()) > 0:
            wedges, texts, autotexts = ax1.pie(trade_distribution.values(),
                                               labels=trade_distribution.keys(),
                                               autopct="%1.1f%%",
                                               startangle=90,
                                               colors=["#4CAF50", "#FF9800"],
                                               explode=(0.05, 0) if metrics.p2p_trade_ratio >= 0.5 else (0, 0.05))
        else:
            ax1.text(0.5,
                     0.5,
                     "No trading data available",
                     ha="center",
                     va="center",
                     transform=ax1.transAxes,
                     fontsize=12,
                     color=self.colors["text"])
            texts = []
            autotexts = []

        # Make text more readable
        for text in texts:
            text.set_fontsize(12)

        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_weight("bold")

        ax1.set_title("Energy Trade Distribution",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"])

        # 2. DSO trade breakdown (buy vs sell volumes)
        ax2 = fig.add_subplot(gs[0, 1])

        dso_breakdown = {"DSO Purchases (FIT)": metrics.dso_buy_volume,
                         "DSO Sales (Utility)": metrics.dso_sell_volume}

        bars = ax2.bar(dso_breakdown.keys(),
                       dso_breakdown.values(),
                       color=["#2196F3", "#E91E63"],
                       width=0.6)

        # Add volume values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.,
                     height*1.01,
                     f"{height:.2f}",
                     ha="center",
                     va="bottom",
                     fontsize=10)

        ax2.set_title("DSO Trade Volumes", fontsize=14, weight="bold")
        ax2.set_ylabel("Energy Volume")
        plt.xticks(rotation=15)

        # 3. Price comparison (local vs DSO buy vs DSO sell)
        ax3 = fig.add_subplot(gs[1, 0])
        price_data = {"Local Market Avg": metrics.local_price_avg,
                      "DSO Buy (FIT)": metrics.dso_buy_price_avg,
                      "DSO Sell (Utility)": metrics.dso_sell_price_avg}

        bars = ax3.bar(price_data.keys(),
                       price_data.values(),
                       color=["#4CAF50", "#2196F3", "#E91E63"],
                       width=0.6)

        # Add price values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2.,
                     height*1.01,
                     f"{height:.4f}",
                     ha="center",
                     va="bottom",
                     fontsize=10)

        ax3.set_title("Price Comparison", fontsize=14, weight="bold")
        ax3.set_ylabel("Price ($/kWh)")
        plt.xticks(rotation=15)

        # 4. Market metrics (decentralization, DSO dependency)
        ax4 = fig.add_subplot(gs[1, 1])
        market_metrics = {"Market Decentralization": metrics.market_decentralization,
                          "DSO Dependency Index": metrics.dso_dependency_index}
        # Use colors based on decentralization level
        decentralization_color = "#4CAF50" if metrics.market_decentralization > 0.7 else "#FFC107" if metrics.market_decentralization > 0.3 else "#F44336"
        dependency_color = "#4CAF50" if metrics.dso_dependency_index < 0.3 else "#FFC107" if metrics.dso_dependency_index < 0.7 else "#F44336"

        bars = ax4.bar(list(market_metrics.keys()),
                       list(market_metrics.values()),
                       color=[decentralization_color, dependency_color],
                       width=0.6)

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.,
                     height*1.01,
                     f"{height:.2f}",
                     ha="center",
                     va="bottom",
                     fontsize=10)

        ax4.set_title("Market Characteristics", fontsize=14, weight="bold")
        ax4.set_ylabel("Index Value (0-1)")
        ax4.set_ylim(0, 1.1)
        plt.xticks(rotation=15)

        # 5. Time series of trade volumes
        ax5 = fig.add_subplot(gs[2, :])
        time_periods = range(len(metrics.dso_trades_over_time))

        ax5.plot(time_periods,
                 metrics.local_trades_over_time,
                 label="Local Trades",
                 color="#4CAF50",
                 linewidth=2)

        ax5.plot(time_periods,
                 metrics.dso_trades_over_time,
                 label="DSO Trades",
                 color="#FFC107",
                 linewidth=2)

        # Add DSO ratio as a secondary axis
        ax5_2 = ax5.twinx()
        ax5_2.plot(time_periods,
                   np.array(metrics.dso_ratio_over_time) * 100,
                   label="DSO Ratio (%)",
                   color="#F44336",
                   linestyle="--",
                   linewidth=1.5)

        # Set labels and title
        ax5.set_xlabel("Time Period")
        ax5.set_ylabel("Trade Volume")
        ax5_2.set_ylabel("DSO Ratio (%)")
        ax5.set_title("Trade Volumes Over Time", fontsize=14, weight="bold")

        # Combine legends
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_2.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # Add grid for readability
        ax5.grid(True, alpha=0.3)

        # Add text summary with key metrics
        plt.figtext(0.5,
                    0.01,
                    f"Net Grid Import: {metrics.net_grid_import:.2f} kWh | "
                    f"Price Spread: {metrics.price_spread:.4f} $/kWh | "
                    f"Local Price Advantage: {metrics.local_price_advantage:.4f} $/kWh | "
                    f"Avoided DSO Cost: ${metrics.avoided_dso_cost:.2f}",
                    ha="center",
                    fontsize=12,
                    bbox={"facecolor":"#E8EAF6", "alpha":0.7, "pad":5, "boxstyle": "round,pad=0.5"})
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        self._save_figure("dso_metrics")

        return fig

    def plot_agent_behavior(self, agent_metrics: Dict[str, List[float]]) -> Figure:
        """Plot individual agent behavior patterns.

        Args:
            agent_metrics: Dictionary of agent metrics over time

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.colors["background"])

        # Define a color palette inspired by the reference figures
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Plot trading activity
        for i, (agent_id, values) in enumerate(agent_metrics.items()):
            color = colors[i % len(colors)]
            ax.plot(values,
                    label=agent_id,
                    color=color,
                    linewidth=2.5,
                    marker='o',
                    markersize=4,
                    markerfacecolor='white',
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    alpha=0.9)

        ax.set_title("Agent Trading Activity",
                     fontsize=16,
                     fontweight="bold",
                     color=self.colors["title"],
                     pad=20)
        ax.set_ylabel("Activity Level", fontsize=12, color=self.colors["text"])
        ax.set_xlabel("Time", fontsize=12, color=self.colors["text"])

        # Legend
        ax.legend(loc="best",
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  fontsize=10,
                  title="Agents",
                  title_fontsize=11)

        # Add subtle grid
        ax.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors["text"])
        ax.spines['bottom'].set_color(self.colors["text"])

        # Set tick colors
        ax.tick_params(colors=self.colors["text"])

        plt.tight_layout()

        self._save_figure("agent_behavior")

        return fig

    def create_market_performance_figure(self, matching_history: MatchingHistory) -> Figure:
        """Create a figure showing market performance over time.

        Args:
            matching_history: History of market matching results

        Returns:
            Matplotlib figure
        """
        if not matching_history.history:
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor(self.colors["background"])

            ax.text(0.5,
                    0.5,
                    "No market data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=16,
                    color=self.colors["text"])

            ax.set_title("Market Performance Over Time",
                         fontsize=18,
                         fontweight="bold",
                         color=self.colors["title"],
                         pad=20)

            return fig

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor(self.colors["background"])

        # Define color palette inspired by reference figures
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Extract time series data
        steps = list(range(len(matching_history.history)))
        clearing_prices = [r.clearing_price for r in matching_history.history]
        clearing_volumes = [r.clearing_volume for r in matching_history.history]
        p2p_volumes = [r.p2p_volume for r in matching_history.history]
        dso_volumes = [r.dso_total_volume for r in matching_history.history]

        # Price over time
        ax1.plot(steps,
                 clearing_prices,
                 color=colors[0],
                 linewidth=3,
                 marker='o',
                 markersize=6,
                 markerfacecolor='white',
                 markeredgecolor=colors[0],
                 markeredgewidth=2,
                 alpha=0.9)

        ax1.set_title("Clearing Price Over Time",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"],
                      pad=15)
        ax1.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
        ax1.set_ylabel("Price ($/kWh)", fontsize=12, color=self.colors["text"])
        ax1.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color(self.colors["text"])
        ax1.spines['bottom'].set_color(self.colors["text"])
        ax1.tick_params(colors=self.colors["text"])

        # Volume over time
        ax2.plot(steps,
                 clearing_volumes,
                 color=colors[2],
                 linewidth=3,
                 marker='o',
                 markersize=6,
                 markerfacecolor='white',
                 markeredgecolor=colors[2],
                 markeredgewidth=2,
                 alpha=0.9,
                 label="Total Volume")

        ax2.plot(steps,
                 p2p_volumes,
                 color=colors[3],
                 linewidth=3,
                 marker='s',
                 markersize=6,
                 markerfacecolor='white',
                 markeredgecolor=colors[3],
                 markeredgewidth=2, alpha=0.9, label="P2P Volume")

        ax2.plot(steps,
                 dso_volumes,
                 color=colors[1],
                 linewidth=3,
                 marker='^',
                 markersize=6, markerfacecolor='white', markeredgecolor=colors[1],
                 markeredgewidth=2,
                 alpha=0.9,
                 label="DSO Volume")

        ax2.set_title("Trading Volume Over Time",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"],
                      pad=15)

        ax2.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
        ax2.set_ylabel("Volume (kWh)", fontsize=12, color=self.colors["text"])
        ax2.legend(loc="best", frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax2.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color(self.colors["text"])
        ax2.spines['bottom'].set_color(self.colors["text"])
        ax2.tick_params(colors=self.colors["text"])

        # P2P ratio over time
        p2p_ratios = [p2p / max(total, 0.01) for p2p, total in zip(p2p_volumes, clearing_volumes)]

        ax3.plot(steps,
                 p2p_ratios,
                 color=colors[4],
                 linewidth=3,
                 marker='D',
                 markersize=6,
                 markerfacecolor='white',
                 markeredgecolor=colors[4],
                 markeredgewidth=2,
                 alpha=0.9)

        ax3.set_title("P2P Trading Ratio Over Time",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"],
                      pad=15)

        ax3.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
        ax3.set_ylabel("P2P Ratio", fontsize=12, color=self.colors["text"])
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_color(self.colors["text"])
        ax3.spines['bottom'].set_color(self.colors["text"])
        ax3.tick_params(colors=self.colors["text"])

        # Trade count over time
        trade_counts = [len(r.trades) for r in matching_history.history]

        bars = ax4.bar(steps,
                       trade_counts,
                       alpha=0.8,
                       color=colors[5],
                       edgecolor='white',
                       linewidth=1.5)

        ax4.set_title("Number of Trades Over Time",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"],
                      pad=15)

        ax4.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
        ax4.set_ylabel("Number of Trades", fontsize=12, color=self.colors["text"])
        ax4.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0, axis='y')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_color(self.colors["text"])
        ax4.spines['bottom'].set_color(self.colors["text"])
        ax4.tick_params(colors=self.colors["text"])

        # Add value labels on bars
        for bar, count in zip(bars, trade_counts):
            if count > 0:
                ax4.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.1,
                         str(count),
                         ha='center',
                         va='bottom',
                         fontweight='bold',
                         fontsize=9)

        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        self._save_figure("market_performance")

        return fig

    def create_coordination_effectiveness_figure(self, matching_history: MatchingHistory) -> Figure:
        """Create a figure showing coordination effectiveness metrics.

        Args:
            matching_history: History of market matching results

        Returns:
            Matplotlib figure
        """
        if not matching_history.history:
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor(self.colors["background"])

            ax.text(0.5,
                    0.5,
                    "No coordination data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=16,
                    color=self.colors["text"])

            ax.set_title("Coordination Effectiveness",
                         fontsize=18,
                         fontweight="bold",
                         color=self.colors["title"],
                         pad=20)
            return fig

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor(self.colors["background"])

        # Define color palette inspired by reference figures
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Extract coordination data
        steps = list(range(len(matching_history.history)))
        grid_balances = [abs(r.grid_balance) for r in matching_history.history]
        p2p_volumes = [r.p2p_volume for r in matching_history.history]
        total_volumes = [r.clearing_volume for r in matching_history.history]

        # Grid balance over time (lower is better)
        ax1.plot(steps,
                 grid_balances,
                 color=colors[3],
                 linewidth=3,
                 marker='o',
                 markersize=6,
                 markerfacecolor='white',
                 markeredgecolor=colors[3],
                 markeredgewidth=2,
                 alpha=0.9)

        ax1.set_title("Grid Balance Deviation Over Time",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"],
                      pad=15)

        ax1.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
        ax1.set_ylabel("Grid Balance Deviation", fontsize=12, color=self.colors["text"])
        ax1.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color(self.colors["text"])
        ax1.spines['bottom'].set_color(self.colors["text"])
        ax1.tick_params(colors=self.colors["text"])

        # P2P coordination ratio
        p2p_ratios = [p2p / max(total, 0.01) for p2p, total in zip(p2p_volumes, total_volumes)]

        ax2.plot(steps,
                 p2p_ratios,
                 color=colors[2],
                 linewidth=3,
                 marker='s',
                 markersize=6,
                 markerfacecolor='white',
                 markeredgecolor=colors[2],
                 markeredgewidth=2,
                 alpha=0.9)

        ax2.set_title("P2P Coordination Ratio",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"],
                      pad=15)

        ax2.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
        ax2.set_ylabel("P2P Ratio", fontsize=12, color=self.colors["text"])
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color(self.colors["text"])
        ax2.spines['bottom'].set_color(self.colors["text"])
        ax2.tick_params(colors=self.colors["text"])

        # Coordination efficiency (combination of grid balance and P2P ratio)
        # Normalize grid balance against max balance across all time steps
        max_balance = max(grid_balances) if grid_balances else 1.0
        coordination_scores = [1 - (balance / max(max_balance, 0.01)) * (1 - ratio) for balance, ratio in zip(grid_balances, p2p_ratios)]

        ax3.plot(steps,
                 coordination_scores,
                 color=colors[4],
                 linewidth=3,
                 marker='D',
                 markersize=6,
                 markerfacecolor='white',
                 markeredgecolor=colors[4],
                 markeredgewidth=2,
                 alpha=0.9)

        ax3.set_title("Coordination Effectiveness Score",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"],
                      pad=15)

        ax3.set_xlabel("Time Step", fontsize=12, color=self.colors["text"])
        ax3.set_ylabel("Coordination Score", fontsize=12, color=self.colors["text"])
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_color(self.colors["text"])
        ax3.spines['bottom'].set_color(self.colors["text"])
        ax3.tick_params(colors=self.colors["text"])

        # Trade distribution
        trade_counts = [len(r.trades) for r in matching_history.history]
        bins = max(1, len(set(trade_counts)))

        n, bins_edges, patches = ax4.hist(trade_counts,
                                          bins=bins,
                                          alpha=0.8,
                                          color=colors[1],
                                          edgecolor='white',
                                          linewidth=1.5)

        ax4.set_title("Distribution of Trade Counts",
                      fontsize=14,
                      fontweight="bold",
                      color=self.colors["title"],
                      pad=15)

        ax4.set_xlabel("Number of Trades", fontsize=12, color=self.colors["text"])
        ax4.set_ylabel("Frequency", fontsize=12, color=self.colors["text"])
        ax4.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0, axis='y')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_color(self.colors["text"])
        ax4.spines['bottom'].set_color(self.colors["text"])
        ax4.tick_params(colors=self.colors["text"])

        # Add value labels on histogram bars
        for i, (patch, count) in enumerate(zip(patches, n)):
            if count > 0:
                ax4.text(patch.get_x() + patch.get_width()/2,
                         patch.get_height() + 0.1,
                         str(int(count)),
                         ha='center',
                         va='bottom',
                         fontweight='bold',
                         fontsize=9)

        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        self._save_figure("coordination_effectiveness")

        return fig

    def plot_sequential_data(self,
                             data: Dict[str, Dict[str, List[float]]],
                             title: str = "Sequential Performance Comparison",
                             x_label: str = "Epoch",
                             y_label: str = "Reward",
                             figsize: tuple = (12, 8)) -> Figure:
        """Create a sequential plot comparing performance across algorithms and approaches.

        This method creates a line plot showing performance over time (e.g., Rewards vs Epoch)
        for multiple RL algorithms with different approaches. The color scheme uses distinct
        base colors for each algorithm with gradients for different approaches within each algorithm.

        Args:
            data: Nested dictionary with structure:
                  {
                      "algorithm_name": {
                          "approach_name": [list_of_values]
                      }
                  }
                  Example: {"ppo": {"ctce": [1,2,3], "ctde": [1,2,3], "dtde": [1,2,3]}}
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            figsize: Figure size tuple

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(self.colors["background"])

        # Define color scheme inspired by the gradient matrix visualization
        # Each algorithm gets a distinct base color with gradients for approaches
        algorithm_colors = {"ppo": "#E53E3E",      # Red base
                            "appo": "#DD6B20",     # Orange base
                            "sac": "#3182CE"}       # Blue base

        # Define approach gradients (dark to light within each algorithm)
        approach_gradients = {"ctce": 1.0,    # Darkest (full saturation)
                              "ctde": 0.7,    # Medium
                              "dtde": 0.4}     # Lightest

        # Line styles for better distinction
        line_styles = {"ctce": "-",     # Solid line
                       "ctde": "--",    # Dashed line
                       "dtde": "-."}     # Dash-dot line

        # Track legend elements
        legend_elements = []

        # Plot each algorithm and approach combination
        for algorithm, approaches in data.items():
            base_color = algorithm_colors.get(algorithm.lower(), "#666666")

            for approach, values in approaches.items():
                if not values:  # Skip empty data
                    continue

                # Calculate gradient color
                gradient_factor = approach_gradients.get(approach.lower(), 0.7)

                # Convert hex to RGB and apply gradient
                hex_color = base_color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                gradient_rgb = tuple(int(c * gradient_factor) for c in rgb)
                gradient_color = f"#{gradient_rgb[0]:02x}{gradient_rgb[1]:02x}{gradient_rgb[2]:02x}"

                # Create x-axis (epochs)
                epochs = list(range(1, len(values) + 1))

                # Plot the line
                line_style = line_styles.get(approach.lower(), "-")

                line = ax.plot(epochs, values,
                               color=gradient_color,
                               linestyle=line_style,
                               linewidth=2.5,
                               marker='o',
                               markersize=4,
                               markerfacecolor='white',
                               markeredgecolor=gradient_color,
                               markeredgewidth=1.5,
                               alpha=0.9,
                               label=f"{algorithm.upper()} - {approach.upper()}")

                # Add to legend
                legend_elements.append(line[0])

        # Customize the plot
        ax.set_title(title, fontsize=16, fontweight="bold", color=self.colors["title"], pad=20)
        ax.set_xlabel(x_label, fontsize=12, color=self.colors["text"])
        ax.set_ylabel(y_label, fontsize=12, color=self.colors["text"])

        # Add grid
        ax.grid(True, alpha=0.3, color=self.colors["grid"], zorder=0)

        # Customize legend
        ax.legend(handles=legend_elements,
                  loc="best",
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  fontsize=10,
                  title="Algorithm - Approach",
                  title_fontsize=11)

        # Set axis properties
        ax.tick_params(colors=self.colors["text"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors["text"])
        ax.spines['bottom'].set_color(self.colors["text"])

        # Adjust layout
        plt.tight_layout()

        # Save if path is provided
        filename = title.lower().replace(" ", "_").replace("-", "_")
        self._save_figure(filename)

        return fig
