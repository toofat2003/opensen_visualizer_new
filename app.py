import streamlit as st
import pandas as pd
import glob
import time
import base64
from datetime import datetime

# baseballmetricsモジュール（成績計算関数群）をインポート
from baseballmetrics import *

# キャッシュ戦略の改善: st.cache_resourceを使用してデータロードを最適化
@st.cache_resource(ttl=86640)
def load_data_from_csv_files():
    """
    ローカルのCSVファイルからデータを効率的に読み込む関数
    
    改善点:
    - 複数ファイルを一度にリスト化してから結合
    - データ型を事前に指定
    - エラーハンドリングの改善
    """
    try:
        start_time = time.time()
        # CSVファイルのパスを取得
        csv_files = glob.glob('data/*.csv')
        
        if not csv_files:
            st.error("データファイルが見つかりません。'data/'ディレクトリにCSVファイルがあることを確認してください。")
            return None
            
        # データ型を事前に指定（日付とよく使う数値列のみ）
        dtypes = {
            'OutsOnPlay': 'Int64',
            'RunsScored': 'Int64',
        }
        parse_dates = ['Date']
        
        # 全てのCSVファイルを一度にリストに読み込み
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(
                    file, 
                    encoding='utf-8-sig',
                    dtype=dtypes,
                    parse_dates=parse_dates,
                    low_memory=False
                )
                dfs.append(df)
            except Exception as e:
                st.warning(f"ファイル '{file}' の読み込み中にエラーが発生しました: {e}")
                continue
                
        if not dfs:
            st.error("有効なCSVファイルがありませんでした。")
            return None
            
        # データフレームをまとめて結合（一度だけconcatを実行）
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # 処理時間のログ
        end_time = time.time()
        st.write(f"データロード時間: {end_time - start_time:.2f}秒")
        
        return combined_df
        
    except Exception as e:
        st.error(f"CSVファイルの読み込みに失敗しました: {e}")
        return None

# 中間処理結果もキャッシュして再計算を防ぐ
@st.cache_data(ttl=86400)
def filter_data(df, level_filter, date_range, side_filter, team_filter, data_type):
    """データのフィルタリング処理をキャッシュする関数"""
    
    # データがない場合は早期リターン
    if df is None or df.empty:
        return None
        
    # Level（A/B)によるフィルタリング
    if level_filter:
        df = df[df["Level"].isin(level_filter)]
    
    # 日付でのフィルタリング
    if date_range:
        df = df[(df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])]
    
    # 投手/打者によるフィルタリング
    if data_type == "打者成績":
        if side_filter:
            df = df[df["PitcherThrows"].isin(side_filter)]
        # チームでフィルタリング
        if team_filter:
            df = df[df["BatterTeam"] == team_filter]
    else:  # 投手成績
        if side_filter:
            df = df[df["BatterSide"].isin(side_filter)]
        # チームでフィルタリング
        if team_filter:
            df = df[df["PitcherTeam"] == team_filter]
    
    return df

# 打者成績計算の最適化
@st.cache_data(ttl=86400)
def compute_batter_stats_optimized(df):
    """
    打者成績を計算する関数（最適化版）
    
    改善点:
    - groupbyとagg関数を活用
    - 不要なデータコピーを削除
    - 処理を関数化して再利用性を向上
    """
    if df is None or df.empty:
        return pd.DataFrame()
        
    # チームフィルターは上流で適用するため、ここでは削除
    # df = df.query('BatterTeam == "TOK"').reset_index(drop=True)
    
    results = []
    # 各打者ごとの統計を計算
    for batter, group in df.groupby("Batter"):
        # 各統計を一度だけ計算
        plate_appearances = seki(group)
        at_bats = dasu(group)
        
        # 各種ヒット数を計算
        single = countpr(group, 'Single')
        double = countpr(group, 'Double')
        triple = countpr(group, 'Triple')
        homerun = countpr(group, 'HomeRun')
        hits = single + double + triple + homerun
        
        # 四球と三振
        walks = countpr(group, 'Walk')
        strikeouts = countpr(group, 'Strikeout')
        
        # 犠打と犠飛
        sac_bunt = countpr(group.query('TaggedHitType == "GroundBall"'), 'Sacrifice')
        sac_fly = countpr(group.query('TaggedHitType == "FlyBall"'), 'Sacrifice')
        
        # 打点
        rbi = int(group.RunsScored.sum())
        
        # 計算された指標
        batting_avg = BA(group)
        on_base_pct = OBP(group, mc=False)
        slugging_pct = SA(group)
        ops = OPS(group)
        
        # 三振率
        so_rate = round(strikeouts / plate_appearances * 100, 1) if plate_appearances > 0 else 0
        
        # 結果を辞書として追加
        stats = {
            "打者": batter,
            "打席": plate_appearances,
            "打数": at_bats,
            "安打": hits,
            "単打": single,
            "二塁打": double,
            "三塁打": triple,
            "本塁打": homerun,
            "打点": rbi,
            "犠打": sac_bunt,
            "犠飛": sac_fly,
            "四球": walks,
            "三振": strikeouts,
            "打率": batting_avg,
            "出塁率": on_base_pct,
            "長打率": slugging_pct,
            "OPS": ops,
            "三振率": so_rate,
        }
        results.append(stats)
    
    # 結果をDataFrameに変換して並べ替え
    if not results:
        return pd.DataFrame()
        
    result_df = pd.DataFrame(results)
    
    # 表示列の順序を指定
    columns = ["打者", "打率", "打点", "出塁率", "長打率", "OPS", 
               "打席", "打数", "安打", "単打", "二塁打", "三塁打", 
               "本塁打", "四球", "三振", "犠打", "犠飛", "三振率"]
    
    # 存在する列のみを使用して並べ替え
    valid_columns = [col for col in columns if col in result_df.columns]
    result_df = result_df[valid_columns].sort_values("OPS", ascending=False)
    
    return result_df

# 投手成績計算の最適化
@st.cache_data(ttl=86400)
def calculate_stats_pitcher_optimized(df):
    """
    投手成績を計算する関数（最適化版）
    
    改善点:
    - エラー処理を強化
    - 計算の簡略化
    - パフォーマンス最適化
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    def calculate_inning_from_float(num):
        try:
            seisu = int(num)
            shosu = round(num - seisu, 2)
            if shosu < 0.3:
                return str(seisu)
            elif 0.3 <= shosu <= 0.6:
                shosu = '1/3'
            else:
                shosu = '2/3'
            return str(seisu) + ' ' + shosu
        except:
            return '-'

    result = []
    
    # 各投手のデータを処理
    for pitcher_name, pitcher_df in df.groupby('Pitcher'):
        try:
            # 安全なデータ変換
            pitcher_df = pitcher_df.copy()
            
            # アウトカウントと得点の処理（エラーを回避）
            try:
                outs = pitcher_df['OutsOnPlay'].fillna(0).astype(int).sum()
                strikeouts = len(pitcher_df.query('KorBB == "Strikeout"'))
                total_outs = outs + strikeouts
                ini = total_outs / 3
                ini_str = calculate_inning_from_float(ini)
            except Exception as e:
                st.warning(f"投手 {pitcher_name} のイニング計算中にエラー: {e}")
                ini = 0
                ini_str = "-"
            
            # 投球データの処理
            df_pitches = pitcher_df.dropna(subset=['PitchofPA'])
            total_pitches = len(df_pitches) if not df_pitches.empty else 0
            
            # 各種ヒット数
            ph = ['InPlay', 'HitByPitch']
            kb = ['Strikeout', 'Walk']
            df_kekka = pitcher_df.query('PitchCall == @ph or KorBB == @kb')
            
            # 三振と四球
            strikeouts = len(pitcher_df.query('KorBB == "Strikeout"'))
            walks = len(pitcher_df.query('KorBB == "Walk"'))
            
            # 打席数と得点
            plate_appearances = seki(pitcher_df)
            runs = pitcher_df['RunsScored'].fillna(0).astype(int).sum()
            
            # ストライク率
            strike_rate = round(len(df_pitches.query('PitchCall != "BallCalled"')) / total_pitches, 2) if total_pitches > 0 else 0
            
            # K-BB%
            k_bb_pct = round((strikeouts - walks) / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            
            # 被安打率とOPS
            batting_avg = BA(pitcher_df, mc=False)
            ops = OPS(pitcher_df)
            
            # K%, BB%, 打球分布
            k_pct = round(strikeouts / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            bb_pct = round(walks / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            
            gb_pct = round(len(pitcher_df.query('TaggedHitType == "GroundBall" and PitchCall == "InPlay"')) / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            ld_pct = round(len(pitcher_df.query('TaggedHitType == "LineDrive" and PitchCall == "InPlay"')) / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            fb_pct = round(len(pitcher_df.query('TaggedHitType == "FlyBall" and PitchCall == "InPlay"')) / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            
            # 走者数を計算
            num_runner = (countpr(pitcher_df, 'Single') + countpr(pitcher_df, 'Double') + 
                          countpr(pitcher_df, 'Triple') + countpr(pitcher_df, 'HomeRun') + 
                          countpr(pitcher_df, 'Walk') + countpr(pitcher_df, 'HitByPitch'))
            
            # WHIP
            whip = round(num_runner / ini, 2) if ini > 0 else 0
            
            # 結果を配列に追加
            pitcher_stats = [
                pitcher_name, ini_str, plate_appearances, runs, 
                BA(pitcher_df, mc=True)[2], strikeouts, walks, strike_rate, 
                k_bb_pct, batting_avg, ops, k_pct, bb_pct, 
                gb_pct, ld_pct, fb_pct, whip
            ]
            
            result.append(pitcher_stats)
            
        except Exception as e:
            st.warning(f"投手 {pitcher_name} の成績計算中にエラー: {e}")
            continue
    
    # 結果をDataFrameに変換
    if not result:
        return pd.DataFrame()
        
    columns = ['Pitcher', '回', '打者', '点', '安', 'K', 'BB', 'ストライク率',
              'K-BB%', '被打率', '被OPS', 'K%', 'BB%', 'GB%', 'LD%', 'FB%', 'WHIP']
    
    result_df = pd.DataFrame(result, columns=columns)
    return result_df

def main():
    # アプリのタイトルと説明
    st.title("オープン戦成績")
    
    # サイドバーの設定
    st.sidebar.title("設定")
    
    # データの読み込み（改善: キャッシュ戦略の変更）
    with st.spinner("データを読み込んでいます..."):
        start_time = time.time()
        
        # 更新ボタン
        update_button = st.sidebar.button("データ更新")
        if update_button:
            st.info("データを更新しています...")
            # キャッシュをクリア
            load_data_from_csv_files.clear()
            filter_data.clear()
            compute_batter_stats_optimized.clear()
            calculate_stats_pitcher_optimized.clear()
        
        # データ読み込み
        df = load_data_from_csv_files()
        
        end_time = time.time()
        load_time = end_time - start_time
        
        # データロード時間の表示
        if df is not None:
            st.write(f"現在の対象試合:2025春 (データロード時間: {load_time:.2f}秒)")
            st.write(f"総レコード数: {len(df)}")
        else:
            st.error("データが取得できませんでした。")
            return
    
    # 説明文
    with st.expander("使い方"):
        st.write("左のサイドバーで、データ種別（打者成績 or 投手成績）やGameLevel（A戦 or B戦）を選択してください。")
        st.write("また、日付範囲をスライダーで選択することで、指定した日付範囲のデータを表示できます。")
    
    # データ種別選択
    type_choice = st.sidebar.radio("データ種別を選択", ["打者成績", "投手成績"])
    
    # GameLevelフィルタ
    st.sidebar.subheader("GameLevel")
    level_filter = []
    if st.sidebar.checkbox("A", value=True):
        level_filter.append("A")
    if st.sidebar.checkbox("B", value=True):
        level_filter.append("B")
    
    if not level_filter:
        st.warning("少なくとも一方のGameLevel（A戦またはB戦）を選択してください。")
        return
    
    # 日付フィルタ
    if df is not None and not df.empty and "Date" in df.columns:
        try:
            min_date = df["Date"].min().to_pydatetime()
            max_date = df["Date"].max().to_pydatetime()
            selected_date_range = st.sidebar.date_input(
                "日付範囲を選択", 
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # 日付選択が単一の場合は範囲として扱う
            if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
                date_range = selected_date_range
            else:
                date_range = (selected_date_range, selected_date_range)
        except Exception as e:
            st.warning(f"日付の処理中にエラーが発生しました: {e}")
            date_range = None
    else:
        date_range = None
    
    # 投手/打者フィルタ
    side_filter = []
    
    if type_choice == "打者成績":
        st.sidebar.subheader("PitcherThrows")
        if st.sidebar.checkbox("Right", value=True):
            side_filter.append("Right")
        if st.sidebar.checkbox("Left", value=True):
            side_filter.append("Left")
    else:  # 投手成績
        st.sidebar.subheader("BatterSide")
        if st.sidebar.checkbox("Right", value=True):
            side_filter.append("Right")
        if st.sidebar.checkbox("Left", value=True):
            side_filter.append("Left")
    
    if not side_filter:
        st.warning("少なくとも一方の選択（RightまたはLeft）を選択してください。")
        return
    
    # チームフィルタ（常にTOKに固定）
    team_filter = "TOK"  
    
    # データのフィルタリング（キャッシュ付き）
    with st.spinner("データをフィルタリング中..."):
        filtered_df = filter_data(df, level_filter, date_range, side_filter, team_filter, type_choice)
    
    if filtered_df is None or filtered_df.empty:
        st.warning("該当するデータがありません。フィルタ条件を変更してください。")
        return
    
    # 成績計算と表示
    with st.spinner("成績を計算中..."):
        if type_choice == "打者成績":
            try:
                stats_df = compute_batter_stats_optimized(filtered_df)
                if stats_df.empty:
                    st.warning("該当する打者データがありません。")
                    return
                    
                st.subheader("打者成績")
                
                # 表示オプション
                use_pagination = st.checkbox("ページネーションを使用", value=True)
                
                if use_pagination:
                    # ページごとの表示件数
                    rows_per_page = st.slider("1ページあたりの表示件数", 5, 50, 10)
                    
                    # データをページ分割して表示
                    total_rows = len(stats_df)
                    total_pages = (total_rows + rows_per_page - 1) // rows_per_page
                    
                    if total_pages > 1:
                        page = st.selectbox("ページ", range(1, total_pages + 1))
                        start_idx = (page - 1) * rows_per_page
                        end_idx = min(start_idx + rows_per_page, total_rows)
                        
                        # 現在のページのデータを表示
                        st.dataframe(stats_df.iloc[start_idx:end_idx], use_container_width=True)
                        st.write(f"表示: {start_idx+1}～{end_idx} / 全{total_rows}件")
                    else:
                        st.dataframe(stats_df, use_container_width=True)
                else:
                    # ページネーションなしで全データを表示
                    st.dataframe(stats_df, use_container_width=True)
                    
                # CSV出力オプション
                if st.button("CSVでダウンロード"):
                    csv = stats_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="batter_stats.csv">CSVファイルをダウンロード</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error("打者成績の集計に失敗しました")
                st.exception(e)
        
        else:  # 投手成績
            try:
                stats_df = calculate_stats_pitcher_optimized(filtered_df)
                if stats_df.empty:
                    st.warning("該当する投手データがありません。")
                    return
                
                # 並び替え
                sort_column = st.selectbox(
                    "並び替え項目", 
                    ['K-BB%', 'K%', 'BB%', 'WHIP', '被打率', '被OPS'],
                    index=0
                )
                stats_df = stats_df.sort_values(sort_column, ascending=False)
                
                st.subheader("投手成績")
                
                # 表示オプション
                use_pagination = st.checkbox("ページネーションを使用", value=True)
                
                if use_pagination:
                    # ページごとの表示件数
                    rows_per_page = st.slider("1ページあたりの表示件数", 5, 50, 10)
                    
                    # データをページ分割して表示
                    total_rows = len(stats_df)
                    total_pages = (total_rows + rows_per_page - 1) // rows_per_page
                    
                    if total_pages > 1:
                        page = st.selectbox("ページ", range(1, total_pages + 1))
                        start_idx = (page - 1) * rows_per_page
                        end_idx = min(start_idx + rows_per_page, total_rows)
                        
                        # 現在のページのデータを表示
                        st.dataframe(stats_df.iloc[start_idx:end_idx], use_container_width=True)
                        st.write(f"表示: {start_idx+1}～{end_idx} / 全{total_rows}件")
                    else:
                        st.dataframe(stats_df, use_container_width=True)
                else:
                    # ページネーションなしで全データを表示
                    st.dataframe(stats_df, use_container_width=True)
                
                # CSV出力オプション
                if st.button("CSVでダウンロード"):
                    csv = stats_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="pitcher_stats.csv">CSVファイルをダウンロード</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error("投手成績の集計に失敗しました")
                st.exception(e)

# プロファイリング機能の追加（オプション）
def add_profiling():
    """
    アプリにプロファイリング機能を追加する関数
    
    この関数は開発時のみ有効にして、本番環境ではコメントアウトすることを推奨
    """
    try:
        import cProfile
        import pstats
        import io
        from contextlib import contextmanager
        
        @contextmanager
        def profiled():
            pr = cProfile.Profile()
            pr.enable()
            yield
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # 上位20件のみ表示
            st.code(s.getvalue())
        
        # プロファイリングを有効にするかのチェックボックス
        if st.sidebar.checkbox("パフォーマンス分析を有効化"):
            with profiled():
                main()
        else:
            main()
    except ImportError:
        # プロファイリングライブラリが利用できない場合は通常実行
        main()

if __name__ == "__main__":
    # プロファイリングを有効にする場合はコメントを外す
    # add_profiling()
    
    # 通常実行
    main()