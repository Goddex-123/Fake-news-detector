import streamlit as st
from dashboard.dashboard_components import create_network_graph

def render_network_analysis(data):
    network_graph = data['network_graph']
    accounts_df = data['accounts_df']
    social_df = data['social_df']
    
    st.markdown("## 🕸️ Network Analysis")
    
    if network_graph is None:
        st.warning("Network graph not yet built. Run the network analyzer:")
        st.code("python src/network_analysis/network_detector.py", language="bash")
    else:
        st.markdown(f"**Network Statistics:** {network_graph.number_of_nodes()} nodes, {network_graph.number_of_edges()} edges")
        
        # Network visualization
        st.markdown("### Network Graph")
        st.info("🔴 Red nodes = Suspicious/Coordinated accounts | 🔵 Blue nodes = Regular accounts")
        
        # For demonstration, mark coordinated accounts as suspicious
        if accounts_df is not None:
            coordinated_ids = accounts_df[accounts_df['coordinated'] == True]['account_id'].tolist()
            suspicious_groups = [{'members': coordinated_ids}]
        else:
            suspicious_groups = None
        
        fig = create_network_graph(network_graph, suspicious_groups=suspicious_groups, max_nodes=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Community statistics
        if social_df is not None and accounts_df is not None:
            st.markdown("### 🚨 Suspicious Activity Detected")
            
            coordinated_accounts = accounts_df[accounts_df['coordinated'] == True]
            coordinated_posts = social_df[social_df['coordinated'] == True]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Coordinated Accounts", len(coordinated_accounts))
            
            with col2:
                st.metric("Coordinated Posts", len(coordinated_posts))
            
            with col3:
                num_groups = len(accounts_df[accounts_df['coordinated'] == True]['group_id'].unique())
                st.metric("Detected Groups", num_groups)
            
            # Group details
            st.markdown("### Coordination Patterns")
            
            if len(coordinated_accounts) > 0:
                for group_id in sorted(accounts_df[accounts_df['coordinated'] == True]['group_id'].unique()):
                    with st.expander(f"Group {group_id}"):
                        group_accounts = accounts_df[accounts_df['group_id'] == group_id]
                        group_posts = social_df[social_df['account_id'].isin(group_accounts['account_id'])]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Size:** {len(group_accounts)} accounts")
                            st.write(f"**Posts:** {len(group_posts)} total posts")
                            st.write(f"**Avg Account Age:** {group_accounts['account_age_days'].mean():.0f} days")
                        
                        with col2:
                            st.write(f"**Verified:** {group_accounts['verified'].sum()} accounts")
                            st.write(f"**Avg Followers:** {group_accounts['followers'].mean():.0f}")
            else:
                st.info("No coordinated groups detected.")
