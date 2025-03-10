import streamlit as st
import pandas as pd
import glob
import time
import base64
from datetime import datetime, date

# baseballmetricsモジュール（成績計算関数群）をインポート
from baseballmetrics import *

# キャッシュ戦略の改善: st.cache_resourceを使用してデータロードを最適化
@st.cache_resource(ttl=86640)
def load_data_from_csv_files():
    """
    ローカルのCSVファイルからデータを効率的に読み込む関数
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
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        try:
            # datetime.date型からpandas datetime64に変換
            start_date = pd.Timestamp(date_range[0])
            end_date = pd.Timestamp(date_range[1])
            
            # 比較前にデータが存在し、適切な型であることを確認
            if "Date" in df.columns and not df["Date"].empty:
                # 日付の範囲でフィルタリング
                df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        except Exception as e:
            st.warning(f"日付フィルタリング中にエラーが発生しました: {e}")
    
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

@st.cache_data(ttl=86400)
def compute_batter_stats_optimized(df):
    """
    打者成績を計算する関数（高速化版）- 各指標を一括で効率的に計算
    """
    if df is None or df.empty:
        return pd.DataFrame()
        
    results = []
    
    # 必要な変数を一度だけ定義
    hit_types = ["Single", "Double", "Triple", "HomeRun"]
    ph = ["InPlay", "HitByPitch"]
    kb = ["Strikeout", "Walk"]
    
    # 各打者ごとに処理
    for batter, batter_df in df.groupby("Batter"):
        try:
            # データフレームをリセット
            df1 = batter_df.reset_index(drop=True)
            
            # 基本的なデータフレームのフィルタリングを一度だけ実行
            filtered_df = df1.query("PitchCall == @ph or KorBB == @kb")
            
            # 打席数
            plate_appearances = len(filtered_df)
            
            # 犠牲打数
            sacrifice_count = len(filtered_df.query('PlayResult == "Sacrifice"'))
            bunt_fc_count = len(filtered_df.query('TaggedHitType == "Bunt" and PlayResult == "FieldersChoice"'))
            sac_total = sacrifice_count + bunt_fc_count
            
            # 四死球数
            bb_count = len(filtered_df.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))
            
            # 打数 (打席 - 犠打 - 四死球)
            at_bats = plate_appearances - sac_total - bb_count
            
            # 各種ヒット数
            single_count = len(filtered_df.query('PlayResult == "Single"'))
            double_count = len(filtered_df.query('PlayResult == "Double"'))
            triple_count = len(filtered_df.query('PlayResult == "Triple"'))
            homerun_count = len(filtered_df.query('PlayResult == "HomeRun"'))
            hit_count = single_count + double_count + triple_count + homerun_count
            
            # 打率
            batting_avg = round(hit_count / at_bats, 3) if at_bats > 0 else np.nan
            
            # 三振数
            strikeout_count = len(filtered_df.query('KorBB == "Strikeout"'))
            
            # 犠打と犠飛
            sac_bunt = len(filtered_df.query('PlayResult == "Sacrifice" and TaggedHitType == "GroundBall"'))
            sac_fly = len(filtered_df.query('PlayResult == "Sacrifice" and TaggedHitType == "FlyBall"'))
            
            # 出塁率の計算
            # バント犠打数（犠飛は含まない）
            bunt_sacrifices = len(filtered_df.query('PlayResult == "Sacrifice" and TaggedHitType in ["Bunt", "GroundBall"]'))
            obp_denominator = plate_appearances - bunt_sacrifices
            obp = round((hit_count + bb_count) / obp_denominator, 3) if obp_denominator > 0 else np.nan
            
            # 長打率の計算
            total_bases = single_count + (2 * double_count) + (3 * triple_count) + (4 * homerun_count)
            slugging_pct = round(total_bases / at_bats, 3) if at_bats > 0 else np.nan
            
            # OPS
            ops = round(obp + slugging_pct, 3) if not (np.isnan(obp) or np.isnan(slugging_pct)) else np.nan
            
            # 三振率
            so_rate = round(strikeout_count / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            
            # 打点
            rbi = int(batter_df.RunsScored.fillna(0).sum())
            
            # 結果を辞書として追加
            stats = {
                "打者": batter,
                "打席": plate_appearances,
                "打数": at_bats,
                "安打": hit_count,
                "単打": single_count,
                "二塁打": double_count,
                "三塁打": triple_count,
                "本塁打": homerun_count,
                "打点": rbi,
                "犠打": sac_bunt,
                "犠飛": sac_fly,
                "四球": len(filtered_df.query('KorBB == "Walk"')),
                "三振": strikeout_count,
                "打率": batting_avg,
                "出塁率": obp,
                "長打率": slugging_pct,
                "OPS": ops,
                "三振率": so_rate,
            }
            results.append(stats)
        except Exception as e:
            st.warning(f"打者 {batter} の成績計算中にエラー: {e}")
            continue
    
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
    date_range = None
    if df is not None and not df.empty and "Date" in df.columns:
        try:
            # 日付の最小値と最大値を取得
            min_date = df["Date"].min()
            max_date = df["Date"].max()
            
            # 日付データが有効であれば表示
            if pd.notna(min_date) and pd.notna(max_date):
                min_date = min_date.date()
                max_date = max_date.date()
                
                # 日付選択ウィジェットを表示
                selected_dates = st.sidebar.date_input(
                    "日付範囲を選択", 
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                # 選択された日付を適切に処理
                if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                    date_range = selected_dates
                elif isinstance(selected_dates, date):
                    # 単一の日付が選択された場合は同じ日付をstart/endに設定
                    date_range = (selected_dates, selected_dates)
                else:
                    st.warning("有効な日付範囲を選択してください。")
            else:
                st.warning("データセットに有効な日付情報がありません。")
        except Exception as e:
            st.warning(f"日付の処理中にエラーが発生しました: {e}")
            st.info("日付フィルタを無視して全データを表示します。")
    
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
                
                # 並び替えオプション
                sort_column = st.selectbox(
                    "並び替え項目", 
                    ["OPS", "打率", "本塁打", "打点", "出塁率", "長打率"],
                    index=0
                )
                ascending = st.checkbox("昇順に並べ替え", value=False)
                stats_df = stats_df.sort_values(sort_column, ascending=ascending)
                
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
                    try:
                        csv = stats_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode("utf-8-sig")).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="batter_stats.csv">CSVファイルをダウンロード</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"CSVダウンロード準備中にエラーが発生しました: {e}")
                
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
                ascending = st.checkbox("昇順に並べ替え", value=False)
                stats_df = stats_df.sort_values(sort_column, ascending=ascending)
                
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
                    try:
                        csv = stats_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode("utf-8-sig")).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="pitcher_stats.csv">CSVファイルをダウンロード</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"CSVダウンロード準備中にエラーが発生しました: {e}")
                
            except Exception as e:
                st.error("投手成績の集計に失敗しました")
                st.exception(e)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"アプリケーション実行中にエラーが発生しました: {e}")
        st.info("詳細なエラー情報を確認するには、Streamlitのデバッグモードを有効にしてください。")