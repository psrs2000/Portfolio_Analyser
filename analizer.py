import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_ticker_validity(tickers, add_sa_suffix=True):
    """Testa se os tickers são válidos fazendo uma busca rápida"""
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        # Só adiciona .SA se a opção estiver ativa E o ticker não terminar com .SA
        if add_sa_suffix and not ticker.endswith('.SA') and len(ticker) <= 6:
            formatted_ticker = f"{ticker}.SA"
        else:
            formatted_ticker = ticker
        
        try:
            # Busca apenas 5 dias de dados para teste rápido
            test_data = yf.download(formatted_ticker, period="5d", progress=False)
            if not test_data.empty:
                valid_tickers.append((ticker, formatted_ticker))
            else:
                invalid_tickers.append((ticker, formatted_ticker))
        except:
            invalid_tickers.append((ticker, formatted_ticker))
    
    return valid_tickers, invalid_tickers

def load_portfolio_data(uploaded_file):
    """Carrega dados da carteira do arquivo CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        # Remove espaços em branco dos nomes das colunas
        df.columns = df.columns.str.strip()
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

def process_portfolio_weights(df, weight_column):
    """Processa os pesos da carteira - versão simplificada"""
    try:
        # Copia DataFrame para não modificar o original
        processed_df = df.copy()
        
        # Converte peso para float, tratando vírgula como separador decimal e removendo %
        processed_df[weight_column] = processed_df[weight_column].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
        
        return processed_df[['Ativo', weight_column]].rename(columns={weight_column: 'Peso'})
    
    except Exception as e:
        st.error(f"Erro ao processar pesos: {e}")
        return None

def fetch_stock_data(tickers, start_date, end_date, add_sa_suffix=True):
    """Busca dados históricos dos ativos no Yahoo Finance"""
    try:
        # Formata tickers baseado na opção escolhida
        formatted_tickers = []
        for ticker in tickers:
            if add_sa_suffix and not ticker.endswith('.SA') and len(ticker) <= 6:
                # Adiciona .SA apenas se a opção estiver ativa e o ticker for brasileiro (<=6 chars e sem .SA)
                formatted_tickers.append(f"{ticker}.SA")
            else:
                # Usa o ticker como está (para ações americanas ou quando já tem .SA)
                formatted_tickers.append(ticker)
        
        st.info(f"Buscando dados para: {', '.join(formatted_tickers)}")
        
        # Busca dados no Yahoo Finance
        data = yf.download(formatted_tickers, start=start_date, end=end_date, progress=False)
        
        # Verifica se os dados foram retornados
        if data.empty:
            st.error("Nenhum dado foi retornado pelo Yahoo Finance")
            return None, None
        
        # Trata diferentes estruturas de retorno
        if len(formatted_tickers) == 1:
            # Para um único ativo
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']].copy()
                prices.columns = [formatted_tickers[0]]
            else:
                # Se não há Adj Close, usa Close
                prices = data[['Close']].copy()
                prices.columns = [formatted_tickers[0]]
        else:
            # Para múltiplos ativos
            if 'Adj Close' in data.columns.levels[0]:
                prices = data['Adj Close'].copy()
            else:
                prices = data['Close'].copy()
        
        # Remove colunas com todos os valores NaN
        prices = prices.dropna(axis=1, how='all')
        
        # Verifica se ainda há dados após limpeza
        if prices.empty:
            st.error("Todos os ativos retornaram dados vazios")
            return None, None
        
        # Mostra quais ativos foram encontrados
        found_tickers = list(prices.columns)
        missing_tickers = [t for t in formatted_tickers if t not in found_tickers]
        
        if found_tickers:
            # Mostra códigos originais e formatados
            found_display = []
            for original, formatted in zip(tickers, formatted_tickers):
                if formatted in found_tickers:
                    if original != formatted:
                        found_display.append(f"{original} → {formatted}")
                    else:
                        found_display.append(original)
            st.success(f"Dados encontrados para: {', '.join(found_display)}")
        
        if missing_tickers:
            # Mostra quais códigos não foram encontrados
            missing_display = []
            for original, formatted in zip(tickers, formatted_tickers):
                if formatted in missing_tickers:
                    if original != formatted:
                        missing_display.append(f"{original} → {formatted}")
                    else:
                        missing_display.append(original)
            st.warning(f"Dados não encontrados para: {', '.join(missing_display)}")
        
        return prices, formatted_tickers
        
    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        st.info("💡 Dicas para resolver:")
        st.info("- Verifique se os códigos dos ativos estão corretos")
        st.info("- Tente um período mais recente")
        st.info("- Alguns ativos podem não estar disponíveis no Yahoo Finance")
        st.info("- Para ações brasileiras, use códigos como PETR4 (sem .SA)")
        st.info("- Para ações americanas, use códigos como AAPL, MSFT")
        return None, None

def calculate_portfolio_returns(prices, weights_dict, fill_method='forward', add_sa_suffix=True):
    """
    Calcula retornos da carteira considerando pesos (incluindo negativos) 
    
    Parâmetros:
    - fill_method: 'forward' (preenche com valor anterior), 'drop' (remove NAs), 'interpolate' (interpola)
    - add_sa_suffix: se True, remove .SA dos códigos para buscar no dicionário de pesos
    """
    
    # Alinha pesos com os ativos disponíveis
    aligned_weights = []
    available_tickers = []
    
    for ticker in prices.columns:
        # Se add_sa_suffix=True, remove .SA para buscar no dicionário
        # Se add_sa_suffix=False, usa o ticker como está
        if add_sa_suffix:
            clean_ticker = ticker.replace('.SA', '')
        else:
            clean_ticker = ticker
            
        if clean_ticker in weights_dict:
            aligned_weights.append(weights_dict[clean_ticker])
            available_tickers.append(ticker)
        else:
            st.warning(f"Peso não encontrado para {clean_ticker} (ticker: {ticker})")
    
    if not aligned_weights:
        st.error("Nenhum peso foi encontrado para os ativos disponíveis")
        return None, None, None, None
    
    # Filtra preços apenas para ativos com pesos
    prices_filtered = prices[available_tickers].copy()
    
    # ========== TRATAMENTO INTELIGENTE DE DADOS FALTANTES ==========
    
    # Mostra estatísticas de dados faltantes antes do tratamento
    missing_stats = prices_filtered.isnull().sum()
    if missing_stats.sum() > 0:
        st.info("📊 Dados faltantes detectados:")
        for ticker, missing_count in missing_stats.items():
            if missing_count > 0:
                total_days = len(prices_filtered)
                missing_pct = (missing_count / total_days) * 100
                clean_name = ticker.replace('.SA', '') if add_sa_suffix else ticker
                st.info(f"• {clean_name}: {missing_count} dias ({missing_pct:.1f}%)")
    
    # Aplica o método de preenchimento escolhido
    if fill_method == 'forward':
        # Preenche com valor anterior (forward fill)
        prices_filtered = prices_filtered.fillna(method='ffill')
        # Se ainda há NAs no início, preenche com o primeiro valor válido
        prices_filtered = prices_filtered.fillna(method='bfill')
        st.info("🔧 Método: Preenchimento com valor anterior (forward fill)")
        
    elif fill_method == 'interpolate':
        # Interpolação linear
        prices_filtered = prices_filtered.interpolate(method='linear')
        # Preenche extremidades se necessário
        prices_filtered = prices_filtered.fillna(method='ffill').fillna(method='bfill')
        st.info("🔧 Método: Interpolação linear")
        
    elif fill_method == 'drop':
        # Remove linhas com qualquer NA
        original_len = len(prices_filtered)
        prices_filtered = prices_filtered.dropna()
        removed_days = original_len - len(prices_filtered)
        if removed_days > 0:
            st.info(f"🔧 Método: Remoção de NAs - {removed_days} dias removidos")
    
    # Verifica se ainda há dados suficientes
    if len(prices_filtered) < 2:
        st.error("❌ Dados insuficientes após tratamento de NAs. Tente:")
        st.error("• Período mais longo")
        st.error("• Método de preenchimento diferente")
        st.error("• Verificar se os códigos dos ativos estão corretos")
        return None, None, None, None
    
    # Mostra estatísticas finais
    final_missing = prices_filtered.isnull().sum().sum()
    if final_missing == 0:
        st.success(f"✅ Dados limpos: {len(prices_filtered)} dias úteis sem dados faltantes")
    else:
        st.warning(f"⚠️ Ainda há {final_missing} dados faltantes após tratamento")
    
    # Normaliza pesos
    weights_array = np.array(aligned_weights)
    total_abs_weight = np.sum(np.abs(weights_array))
    if total_abs_weight != 100:
        st.info(f"Soma dos pesos absolutos: {total_abs_weight:.2f}%. Mantendo pesos originais.")
    
    weights_normalized = weights_array / 100
    
    # ========== CÁLCULO DA CARTEIRA ==========
    
    try:
        # 1. Preços base (primeiro dia)
        first_prices = prices_filtered.iloc[0]
        
        # 2. Variações diárias absolutas
        price_changes = prices_filtered.diff().dropna()
        
        # 3. Variações relativas ao preço inicial
        daily_variations = price_changes.div(first_prices, axis=1)
        
        # 4. Variação diária da carteira (aplica pesos)
        portfolio_daily_variations = (daily_variations * weights_normalized).sum(axis=1)
        
        # 5. Calcula valor das cotas
        portfolio_cota_values = pd.Series(index=prices_filtered.index, dtype=float)
        cota_inicial = 1.0
        portfolio_cota_values.iloc[0] = cota_inicial
        
        # 6. Aplica variações acumuladas de forma segura
        cumulative_variations = portfolio_daily_variations.cumsum()
        
        # Método mais robusto: usar reindex para garantir alinhamento
        aligned_cum_var = cumulative_variations.reindex(portfolio_cota_values.index[1:], fill_value=0)
        portfolio_cota_values.iloc[1:] = cota_inicial + aligned_cum_var.values
        
        # 7. Calcula retornos diários
        portfolio_returns = portfolio_cota_values.pct_change().dropna()
        
        # Validação final
        if len(portfolio_cota_values) < 2:
            st.error("Dados insuficientes após cálculos.")
            return None, None, None, None
        
        return portfolio_returns, portfolio_cota_values, available_tickers, weights_normalized
        
    except Exception as e:
        st.error(f"Erro no cálculo da carteira: {e}")
        st.error("Informações de debug:")
        st.error(f"• Tamanho prices_filtered: {len(prices_filtered)}")
        if 'portfolio_daily_variations' in locals():
            st.error(f"• Tamanho portfolio_daily_variations: {len(portfolio_daily_variations)}")
        st.error(f"• Ativos: {available_tickers}")
        return None, None, None, None


def calculate_monthly_returns(portfolio_cota_values):
    """Calcula retornos mensais da carteira baseado no valor das cotas"""
    # Converte índice para datetime se necessário
    portfolio_cota_values.index = pd.to_datetime(portfolio_cota_values.index)
    
    # Remove valores NaN se existirem
    portfolio_cota_values = portfolio_cota_values.dropna()
    
    # Pega o último valor de cada mês
    monthly_values = portfolio_cota_values.groupby([
        portfolio_cota_values.index.year,
        portfolio_cota_values.index.month
    ]).last()
    
    # Calcula retornos mensais - CORRIGIDO para incluir primeiro mês
    monthly_returns = monthly_values.pct_change()
    
    # Para o primeiro mês, calcula retorno em relação à cota inicial (1.0)
    if len(monthly_returns) > 0:
        first_month_idx = monthly_returns.index[0]
        first_month_value = monthly_values.iloc[0]
        monthly_returns.iloc[0] = (first_month_value - 1.0) / 1.0
    
    # Remove apenas valores NaN reais (não o primeiro mês)
    monthly_returns = monthly_returns.dropna()
    
    # Cria índice mais legível
    monthly_returns.index = [f"{year}-{month:02d}" for year, month in monthly_returns.index]
    
    return monthly_returns

def create_monthly_returns_table(portfolio_cota_values):
    """Cria tabela de retornos mensais em formato matriz (anos x meses) - VERSÃO CORRIGIDA"""
    # Converte índice para datetime se necessário
    portfolio_cota_values.index = pd.to_datetime(portfolio_cota_values.index)
    
    # Remove valores NaN se existirem
    portfolio_cota_values = portfolio_cota_values.dropna()
    
    # Pega valores por mês (último dia útil do mês)
    monthly_values = portfolio_cota_values.groupby([
        portfolio_cota_values.index.year,
        portfolio_cota_values.index.month
    ]).last()
    
    # Calcula retornos mensais - CORRIGIDO para incluir primeiro mês
    monthly_returns = monthly_values.pct_change()
    
    # Para o primeiro mês, calcula retorno em relação à cota inicial (1.0)
    if len(monthly_returns) > 0:
        first_month_idx = monthly_returns.index[0]
        first_month_value = monthly_values.iloc[0]
        monthly_returns.iloc[0] = (first_month_value - 1.0) / 1.0
    
    # Remove apenas valores NaN reais (não o primeiro mês)
    monthly_returns = monthly_returns.dropna()
    
    # Cria DataFrame com anos nas linhas e meses nas colunas
    monthly_matrix = []
    years = sorted(set([year for year, month in monthly_returns.index]))
    
    for year in years:
        row = {'Ano': year}
        annual_cumulative = 1.0  # Para calcular retorno anual
        
        for month in range(1, 13):
            if (year, month) in monthly_returns.index:
                monthly_ret = monthly_returns[(year, month)]
                row[f"{month:02d}"] = f"{monthly_ret * 100:.2f}%"
                annual_cumulative *= (1 + monthly_ret)
            else:
                row[f"{month:02d}"] = "-"
        
        # Adiciona retorno anual
        annual_return = (annual_cumulative - 1) * 100
        row['Ano Total'] = f"{annual_return:.2f}%"
        monthly_matrix.append(row)
    
    # Cria DataFrame
    monthly_df = pd.DataFrame(monthly_matrix)
    
    # Renomeia colunas para nomes dos meses
    month_names = {
        '01': 'Jan', '02': 'Fev', '03': 'Mar', '04': 'Abr',
        '05': 'Mai', '06': 'Jun', '07': 'Jul', '08': 'Ago', 
        '09': 'Set', '10': 'Out', '11': 'Nov', '12': 'Dez'
    }
    
    # Renomeia apenas as colunas que existem
    new_columns = {'Ano': 'Ano'}
    for old_col, new_col in month_names.items():
        if old_col in monthly_df.columns:
            new_columns[old_col] = new_col
    
    monthly_df = monthly_df.rename(columns=new_columns)
    
    return monthly_df

def create_portfolio_chart(portfolio_cota_values):
    """Cria gráfico da evolução da carteira baseado no valor das cotas"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_cota_values.index,
        y=portfolio_cota_values.values,
        mode='lines',
        name='Valor da Cota',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title='Evolução da Carteira de Investimentos (Valor da Cota)',
        xaxis_title='Data',
        yaxis_title='Valor da Cota (Base 1.0)',
        template='plotly_white',
        height=500
    )
    
    return fig

def add_missing_data_options():
    """Adiciona opções de tratamento de dados faltantes na sidebar"""
    st.sidebar.subheader("🔧 Tratamento de Dados Faltantes")
    
    fill_method = st.sidebar.selectbox(
        "Método para dados faltantes:",
        options=['forward', 'interpolate', 'drop'],
        format_func=lambda x: {
            'forward': '📈 Forward Fill (usa valor anterior)',
            'interpolate': '📊 Interpolação Linear', 
            'drop': '🗑️ Remover dias com NAs'
        }[x],
        help="""
        • Forward Fill: Preenche com o último preço conhecido (mais conservador)
        • Interpolação: Estima valores intermediários (suaviza dados)
        • Remover: Exclui dias com dados faltantes (pode reduzir período)
        """
    )
    
    return fill_method

def add_ticker_format_options():
    """Adiciona opções de formato dos códigos dos ativos"""
    st.sidebar.subheader("🏷️ Formato dos Códigos")
    
    ticker_format = st.sidebar.radio(
        "Como estão os códigos no seu arquivo?",
        options=['brazilian', 'yahoo_format'],
        format_func=lambda x: {
            'brazilian': '🇧🇷 Códigos Brasileiros (PETR4, VALE3)',
            'yahoo_format': '🌐 Códigos do Yahoo Finance (PETR4.SA, WMT)'
        }[x],
        help="""
        • Códigos Brasileiros: O sistema adicionará .SA automaticamente para ações brasileiras
        • Códigos Yahoo Finance: Use os códigos exatos (PETR4.SA para brasileiras, WMT para americanas)
        """
    )
    
    # Converte para booleano para compatibilidade com o resto do código
    add_sa_suffix = (ticker_format == 'brazilian')
    
    return add_sa_suffix

def main():
    st.set_page_config(
        page_title="Analisador de Carteira",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Analisador de Carteira de Investimentos")
    st.markdown("---")
    
    # Sidebar para configurações
    st.sidebar.header("Configurações")
    
    # Upload do arquivo CSV
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do arquivo CSV da carteira",
        type=['csv'],
        help="Arquivo deve conter colunas 'Ativo' e 'Peso'"
    )
    
    if uploaded_file is not None:
        # Carrega dados da carteira
        portfolio_df = load_portfolio_data(uploaded_file)
        
        if portfolio_df is not None:
            st.sidebar.success("Arquivo carregado com sucesso!")
            
            # Mostra preview dos dados
            with st.sidebar.expander("Preview dos dados"):
                st.dataframe(portfolio_df)
            
            # ========== NOVA SEÇÃO: Configuração de Colunas ==========
            st.sidebar.subheader("⚙️ Configuração das Colunas")
            
            # Detecta formato automaticamente
            colunas_disponiveis = portfolio_df.columns.tolist()
            
            # Verifica se é formato do otimizador (tem múltiplas colunas de peso)
            peso_columns = [col for col in colunas_disponiveis if 'peso' in col.lower() or '%' in col]
            
            if len(peso_columns) > 1:
                st.sidebar.info("🔍 Formato do Otimizador detectado!")
                
                # Escolha da coluna de peso
                weight_column = st.sidebar.selectbox(
                    "Escolha a coluna de peso:",
                    peso_columns,
                    help="Selecione qual coluna de peso usar para análise"
                )
                
                # Processa os pesos
                processed_df = process_portfolio_weights(portfolio_df, weight_column)
                
                if processed_df is not None:
                    portfolio_df = processed_df
                    
                    # Mostra preview dos dados processados
                    with st.sidebar.expander("Preview dos dados processados"):
                        st.dataframe(portfolio_df)
                        
                        # Mostra estatísticas dos pesos
                        total_peso = portfolio_df['Peso'].sum()
                        peso_positivos = portfolio_df[portfolio_df['Peso'] > 0]['Peso'].sum()
                        peso_negativos = portfolio_df[portfolio_df['Peso'] < 0]['Peso'].sum()
                        
                        st.write(f"**Estatísticas dos Pesos:**")
                        st.write(f"• Total: {total_peso:.2f}%")
                        st.write(f"• Posições LONG: {peso_positivos:.2f}%")
                        st.write(f"• Posições SHORT: {peso_negativos:.2f}%")
            
            elif 'Peso' in colunas_disponiveis:
                st.sidebar.info("📊 Formato simples detectado!")
                # Formato simples - apenas processa a coluna Peso existente
                processed_df = process_portfolio_weights(portfolio_df, 'Peso')
                if processed_df is not None:
                    portfolio_df = processed_df
            
            else:
                st.sidebar.error("❌ Nenhuma coluna de peso encontrada!")
                st.sidebar.info("Colunas disponíveis: " + ", ".join(colunas_disponiveis))
                return
            
            # ========== NOVA SEÇÃO: Formato dos Códigos ==========
            add_sa_suffix = add_ticker_format_options()
            
            # Mostra exemplo baseado na escolha
            if add_sa_suffix:
                st.sidebar.info("💡 Exemplo: PETR4 → PETR4.SA, VALE3 → VALE3.SA")
            else:
                st.sidebar.info("💡 Exemplo: PETR4.SA, WMT, AAPL (códigos exatos)")
            
            # Configuração de datas
            st.sidebar.subheader("Período de Análise")
            
            end_date = st.sidebar.date_input(
                "Data Final",
                value=datetime.now().date()
            )
            
            start_date = st.sidebar.date_input(
                "Data Inicial",
                value=datetime.now().date() - timedelta(days=365)
            )
            
            if start_date >= end_date:
                st.sidebar.error("Data inicial deve ser anterior à data final!")
                return

            fill_method = add_missing_data_options()
            
            # Botão para testar tickers
            if st.sidebar.button("🔍 Testar Ativos", help="Verifica se os códigos dos ativos são válidos"):
                with st.spinner("Testando conectividade com Yahoo Finance..."):
                    tickers = portfolio_df['Ativo'].tolist()
                    valid_tickers, invalid_tickers = test_ticker_validity(tickers, add_sa_suffix)
                    
                    if valid_tickers:
                        st.success(f"✅ Ativos válidos ({len(valid_tickers)}):")
                        for original, formatted in valid_tickers:
                            if original != formatted:
                                st.write(f"• {original} → {formatted}")
                            else:
                                st.write(f"• {original}")
                    
                    if invalid_tickers:
                        st.error(f"❌ Ativos não encontrados ({len(invalid_tickers)}):")
                        for original, formatted in invalid_tickers:
                            if original != formatted:
                                st.write(f"• {original} → {formatted}")
                            else:
                                st.write(f"• {original}")
                        
                        st.info("💡 Sugestões:")
                        st.write("- Verifique a grafia dos códigos")
                        st.write("- Alguns ativos podem ter sido deslistados")
                        st.write("- Para ações brasileiras: PETR4, VALE3 (sem .SA)")
                        st.write("- Para ações americanas: AAPL, MSFT, WMT")
            
            st.sidebar.markdown("---")
            
            # Botão para executar análise
            if st.sidebar.button("🚀 Executar Análise", type="primary"):
                
                with st.spinner("Buscando dados e calculando..."):
                    
                    # Prepara dados
                    tickers = portfolio_df['Ativo'].tolist()
                    weights_dict = dict(zip(portfolio_df['Ativo'], portfolio_df['Peso']))
                    
                    st.info(f"Analisando carteira com {len(tickers)} ativos")
                    st.info(f"Período: {start_date} até {end_date}")
                    
                    # Mostra informação sobre formato dos códigos
                    if add_sa_suffix:
                        st.info("🏷️ Modo: Códigos brasileiros (adicionando .SA automaticamente)")
                    else:
                        st.info("🏷️ Modo: Códigos no formato Yahoo Finance")
                    
                    # Busca dados do Yahoo Finance
                    prices, formatted_tickers = fetch_stock_data(tickers, start_date, end_date, add_sa_suffix)
                    
                    if prices is not None and not prices.empty:
                        # Calcula retornos da carteira
                        portfolio_returns, portfolio_cota_values, available_tickers, weights_normalized = calculate_portfolio_returns(prices, weights_dict, fill_method, add_sa_suffix)
                        
                        if portfolio_returns is not None:
                            
                            # Layout em colunas
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Gráfico da evolução
                                st.subheader("📊 Evolução da Carteira")
                                chart = create_portfolio_chart(portfolio_cota_values)
                                st.plotly_chart(chart, use_container_width=True)
                            
                            with col2:
                                # Métricas principais
                                st.subheader("📈 Métricas")
                                
                                # Cálculos corrigidos baseados no valor da cota
                                total_return = (portfolio_cota_values.iloc[-1] / portfolio_cota_values.iloc[0] - 1) * 100
                                annual_return = ((portfolio_cota_values.iloc[-1] / portfolio_cota_values.iloc[0]) ** (252 / len(portfolio_cota_values)) - 1) * 100
                                volatility = portfolio_returns.std() * np.sqrt(252) * 100
                                sharpe = (annual_return - 5) / volatility if volatility > 0 else 0  # Assumindo taxa livre de risco de 5%
                                
                                st.metric("Retorno Total", f"{total_return:.2f}%")
                                st.metric("Retorno Anualizado", f"{annual_return:.2f}%")
                                st.metric("Volatilidade Anual", f"{volatility:.2f}%")
                                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                                
                                # Composição da carteira
                                st.subheader("🏗️ Composição")
                                composition_df = pd.DataFrame({
                                    'Ativo': [ticker.replace('.SA', '') if add_sa_suffix else ticker for ticker in available_tickers],
                                    'Peso (%)': [w * 100 for w in weights_normalized]
                                })
                                
                                # Mostra soma real dos pesos
                                total_weight_check = sum([w * 100 for w in weights_normalized])
                                st.info(f"Soma real dos pesos: {total_weight_check:.2f}%")
                                
                                st.dataframe(composition_df, hide_index=True)
                            
                            # Tabela de retornos mensais
                            st.subheader("📅 Retornos Mensais")
                            
                            # Cria tabela em formato matriz (anos x meses)
                            monthly_matrix = create_monthly_returns_table(portfolio_cota_values)
                            
                            # Aplica formatação com cores usando CSS
                            def color_negative_red(val):
                                """Aplica cor vermelha para valores negativos e verde para positivos"""
                                if isinstance(val, str) and val != '-':
                                    try:
                                        # Remove o % e converte para float
                                        num_val = float(val.replace('%', ''))
                                        if num_val < 0:
                                            return 'color: #d32f2f; font-weight: bold'  # Vermelho
                                        elif num_val > 0:
                                            return 'color: #388e3c; font-weight: bold'  # Verde
                                        else:
                                            return 'color: #666; font-weight: bold'     # Cinza para zero
                                    except:
                                        return ''
                                return ''
                            
                            # Aplica o estilo apenas nas colunas de meses (exclui 'Ano')
                            monthly_columns = [col for col in monthly_matrix.columns if col != 'Ano']
                            styled_table = monthly_matrix.style.applymap(color_negative_red, subset=monthly_columns)
                            
                            # Mostra a tabela estilizada
                            st.dataframe(styled_table, hide_index=True, use_container_width=True)
                            
                            # Estatísticas dos retornos mensais
                            monthly_returns = calculate_monthly_returns(portfolio_cota_values)
                            st.subheader("📊 Estatísticas dos Retornos Mensais")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Retorno Médio Mensal", f"{monthly_returns.mean() * 100:.2f}%")
                            with col2:
                                st.metric("Melhor Mês", f"{monthly_returns.max() * 100:.2f}%")
                            with col3:
                                st.metric("Pior Mês", f"{monthly_returns.min() * 100:.2f}%")
                            with col4:
                                st.metric("Volatilidade Mensal", f"{monthly_returns.std() * 100:.2f}%")
                        
                        else:
                            st.error("Erro no cálculo dos retornos da carteira")
                    else:
                        st.error("Erro ao buscar dados dos ativos")
    
    else:
        st.info("👆 Faça upload do arquivo CSV da carteira para começar a análise")
        
        # Exemplo de formato
        st.subheader("📋 Formatos de Arquivo CSV Aceitos")
        
        # Formato simples
        st.markdown("**1. Formato Simples:**")
        example_df_simple = pd.DataFrame({
            'Ativo': ['BBAS3', 'BBDC4', 'BBSE3', 'BMEB4'],
            'Peso': ['3.97%', '0.68%', '7.30%', '1.15%']
        })
        st.dataframe(example_df_simple, hide_index=True)
        
        # Formato do otimizador
        st.markdown("**2. Formato do Otimizador:**")
        example_df_optimizer = pd.DataFrame({
            'Ativo': ['BOVA11', 'FIND11', 'PETR4', 'FRAS3'],
            'Peso Inicial (%)': ['-100.00%', '23.15%', '12.12%', '7.25%'],
            'Peso Atual (%)': ['-63.12%', '17.18%', '8.22%', '10.65%'],
            'Tipo': ['SHORT', 'LONG', 'LONG', 'LONG']
        })
        st.dataframe(example_df_optimizer, hide_index=True)
        
        # NOVO: Formato com códigos mistos
        st.markdown("**3. Formato com Códigos Mistos (Brasileiros + Americanos):**")
        example_df_mixed = pd.DataFrame({
            'Ativo': ['WMT', 'BSX', 'EMBR3.SA', 'PLTR', 'CMIG4.SA'],
            'Peso Inicial (%)': ['22.12%', '14.34%', '7.12%', '5.29%', '5.23%'],
            'Peso Atual (%)': ['19.78%', '11.51%', '9.32%', '12.92%', '3.85%'],
            'Tipo': ['LONG', 'LONG', 'LONG', 'LONG', 'LONG']
        })
        st.dataframe(example_df_mixed, hide_index=True)
        
        st.markdown("""
        **Observações importantes:**
        - A coluna 'Ativo' deve conter os códigos das ações
        
        **Para códigos brasileiros:** 
        - Use PETR4, VALE3, BBAS3 (sem .SA)
        - O sistema adicionará .SA automaticamente
        
        **Para códigos no formato Yahoo Finance:**
        - Brasileiras: PETR4.SA, VALE3.SA
        - Americanas: WMT, AAPL, MSFT
        - Use os códigos exatos como aparecem no Yahoo Finance
        
        **Para formato simples:** use coluna 'Peso' com valores percentuais ou decimais
        
        **Para formato do otimizador:** 
        - Você pode escolher entre "Peso Inicial (%)" ou "Peso Atual (%)"
        - A coluna 'Tipo' converte automaticamente SHORT para pesos negativos
        - Pesos negativos representam posições vendidas/short
        
        - Use vírgula ou ponto como separador decimal
        """)        

if __name__ == "__main__":
    main()