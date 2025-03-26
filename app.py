import streamlit as st
import pandas as pd
import glob
import time
import base64
from datetime import datetime, date
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
import os

# baseballmetricsモジュール（成績計算関数群）をインポート
from baseballmetrics import *
# ワイルドカードの代わりに明示的なインポート
from baseballmetrics import calculate_runvalue1_vectorized_final
# --- マルチページ設定 ---
PAGES = {
    "チーム成績": "team_stats",
    "選手成績": "player_stats"
}

def main():
    # アプリのタイトル
    st.title("オープン戦成績")
    
    # ページ選択（サイドバーに配置）
    st.sidebar.title("ナビゲーション")
    page = st.sidebar.radio("ページを選択", list(PAGES.keys()))
    
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
    
    # ページの表示
    if page == "チーム成績":
        team_stats_page(df)
    elif page == "選手成績":
        player_stats_page(df)

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
    打者成績を計算する関数（超高速化版）- ベクトル化処理とクエリの最小化による最適化
    盗塁と盗塁死の統計を追加
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # 最初に必要な列だけを抽出してコピーを作成（メモリ使用量削減）
    needed_columns = ['Batter', 'PitchCall', 'KorBB', 'PlayResult', 'TaggedHitType', 
                      'RunsScored', 'ExitSpeed', 'on_1b', 'on_2b', 'runevent']
    # 存在する列だけを選択
    available_cols = [col for col in needed_columns if col in df.columns]
    df_slim = df[available_cols].copy()
    
    # データフレームに必要な条件フラグを一度に追加（ベクトル化処理）
    # 打席対象のピッチか
    ph = ["InPlay", "HitByPitch"]
    kb = ["Strikeout", "Walk"]
    
    # 条件マスクを作成（ベクトル化）
    df_slim['is_plate_appearance'] = (df_slim['PitchCall'].isin(ph) | df_slim['KorBB'].isin(kb)) if 'PitchCall' in df_slim.columns and 'KorBB' in df_slim.columns else False
    df_slim['is_hit'] = df_slim['PlayResult'].isin(["Single", "Double", "Triple", "HomeRun"]) if 'PlayResult' in df_slim.columns else False
    df_slim['is_sacrifice'] = df_slim['PlayResult'] == "Sacrifice" if 'PlayResult' in df_slim.columns else False
    df_slim['is_bunt_fc'] = ((df_slim['TaggedHitType'] == "Bunt") & (df_slim['PlayResult'] == "FieldersChoice")) if 'TaggedHitType' in df_slim.columns and 'PlayResult' in df_slim.columns else False
    
    # 四死球とヒットバイピッチを別々に計算
    df_slim['is_walk'] = df_slim['KorBB'] == "Walk" if 'KorBB' in df_slim.columns else False
    df_slim['is_hbp'] = df_slim['PitchCall'] == "HitByPitch" if 'PitchCall' in df_slim.columns else False
    df_slim['is_bb_or_hbp'] = df_slim['is_walk'] | df_slim['is_hbp']
    
    df_slim['is_strikeout'] = df_slim['KorBB'] == "Strikeout" if 'KorBB' in df_slim.columns else False
    
    # 各ヒットタイプのフラグ
    df_slim['is_single'] = df_slim['PlayResult'] == "Single" if 'PlayResult' in df_slim.columns else False
    df_slim['is_double'] = df_slim['PlayResult'] == "Double" if 'PlayResult' in df_slim.columns else False
    df_slim['is_triple'] = df_slim['PlayResult'] == "Triple" if 'PlayResult' in df_slim.columns else False
    df_slim['is_homerun'] = df_slim['PlayResult'] == "HomeRun" if 'PlayResult' in df_slim.columns else False
    
    # 犠打と犠飛
    if 'TaggedHitType' in df_slim.columns:
        df_slim['is_sac_bunt'] = df_slim['is_sacrifice'] & (df_slim['TaggedHitType'] == "Bunt")
        df_slim['is_sac_fly'] = df_slim['is_sacrifice'] & (df_slim['TaggedHitType'] == "FlyBall")
    else:
        df_slim['is_sac_bunt'] = False
        df_slim['is_sac_fly'] = False
    
    # 打席のあるデータだけに絞る
    plate_app_df = df_slim[df_slim['is_plate_appearance']]
    
    # 空のデータフレームの場合は、空の結果を返す
    if plate_app_df.empty:
        return pd.DataFrame()
    
    # 必要な統計を一度にまとめて集計（groupby操作の回数を最小化）
    stats = plate_app_df.groupby('Batter').agg(
        plate_appearances=('is_plate_appearance', 'sum'),
        sacrifice_count=('is_sacrifice', 'sum'),
        bunt_fc_count=('is_bunt_fc', 'sum'),
        walk_count=('is_walk', 'sum'),
        hbp_count=('is_hbp', 'sum'),
        single_count=('is_single', 'sum'),
        double_count=('is_double', 'sum'),
        triple_count=('is_triple', 'sum'),
        homerun_count=('is_homerun', 'sum'),
        strikeout_count=('is_strikeout', 'sum'),
        sac_bunt_count=('is_sac_bunt', 'sum'),
        sac_fly_count=('is_sac_fly', 'sum'),
        hit_count=('is_hit', 'sum'),
    ).reset_index()
    
    # 打点の計算 - 別途集計する必要がある
    if 'RunsScored' in df_slim.columns:
        rbi_by_batter = df.groupby('Batter')['RunsScored'].sum().fillna(0).astype(int).reset_index()
        stats = pd.merge(stats, rbi_by_batter, on='Batter', how='left')
    else:
        stats['RunsScored'] = 0
    
    # 盗塁と盗塁死の統計を計算
    # 盗塁と盗塁死のデータを集計するための条件
    if all(col in df.columns for col in ['on_1b', 'on_2b', 'runevent']):
        # 盗塁成功 - 選手がon_1bまたはon_2bにいて、runeventがStolenBase
        # on_1bの選手の盗塁
        sb_1b = df[df['runevent'] == 'StolenBase'].groupby('on_1b').size().reset_index(name='sb_from_1b')
        sb_1b = sb_1b[sb_1b['on_1b'] != '']  # 空の文字列を除外
        
        # on_2bの選手の盗塁
        sb_2b = df[df['runevent'] == 'StolenBase'].groupby('on_2b').size().reset_index(name='sb_from_2b')
        sb_2b = sb_2b[sb_2b['on_2b'] != '']  # 空の文字列を除外
        
        # 盗塁死 - 選手がon_1bまたはon_2bにいて、runeventがCaughtStealing
        # on_1bの選手の盗塁死
        cs_1b = df[df['runevent'] == 'CaughtStealing'].groupby('on_1b').size().reset_index(name='cs_from_1b')
        cs_1b = cs_1b[cs_1b['on_1b'] != '']  # 空の文字列を除外
        
        # on_2bの選手の盗塁死
        cs_2b = df[df['runevent'] == 'CaughtStealing'].groupby('on_2b').size().reset_index(name='cs_from_2b')
        cs_2b = cs_2b[cs_2b['on_2b'] != '']  # 空の文字列を除外
        
        # 各統計をBatterと結合
        # 1塁からの盗塁
        stats = pd.merge(stats, sb_1b.rename(columns={'on_1b': 'Batter'}), on='Batter', how='left')
        # 2塁からの盗塁
        stats = pd.merge(stats, sb_2b.rename(columns={'on_2b': 'Batter'}), on='Batter', how='left')
        # 1塁からの盗塁死
        stats = pd.merge(stats, cs_1b.rename(columns={'on_1b': 'Batter'}), on='Batter', how='left')
        # 2塁からの盗塁死
        stats = pd.merge(stats, cs_2b.rename(columns={'on_2b': 'Batter'}), on='Batter', how='left')
        
        # NaNを0に置換
        for col in ['sb_from_1b', 'sb_from_2b', 'cs_from_1b', 'cs_from_2b']:
            if col in stats.columns:
                stats[col] = stats[col].fillna(0).astype(int)
            else:
                stats[col] = 0
        
        # 盗塁合計と盗塁死合計を計算
        stats['stolen_bases'] = stats['sb_from_1b'] + stats['sb_from_2b']
        stats['caught_stealing'] = stats['cs_from_1b'] + stats['cs_from_2b']
        
        # 盗塁成功率を計算（盗塁試行が0の場合はNaN）
        stats['sb_attempts'] = stats['stolen_bases'] + stats['caught_stealing']
        stats['sb_success_rate'] = np.where(
            stats['sb_attempts'] > 0,
            (stats['stolen_bases'] / stats['sb_attempts'] * 100).round(1),
            np.nan
        )
    else:
        # 必要な列がない場合は0を設定
        stats['stolen_bases'] = 0
        stats['caught_stealing'] = 0
        stats['sb_success_rate'] = np.nan
    
    # 四死球の合計
    stats['bb_count'] = stats['walk_count'] + stats['hbp_count']
    
    # 打数計算
    stats['sac_total'] = stats['sacrifice_count'] + stats['bunt_fc_count']
    stats['at_bats'] = stats['plate_appearances'] - stats['sac_total'] - stats['bb_count']
    
    # 打率、出塁率、長打率の計算（ベクトル化）
    # 0除算を防ぐマスク
    at_bats_mask = stats['at_bats'] > 0
    obp_denominator_mask = (stats['plate_appearances'] - stats['sac_bunt_count']) > 0
    
    # 打率
    stats['batting_avg'] = np.nan
    stats.loc[at_bats_mask, 'batting_avg'] = (stats.loc[at_bats_mask, 'hit_count'] / 
                                             stats.loc[at_bats_mask, 'at_bats']).round(3)
    
    # 出塁率
    stats['obp_denominator'] = stats['plate_appearances'] - stats['sac_bunt_count']
    stats['on_base_pct'] = np.nan
    stats.loc[obp_denominator_mask, 'on_base_pct'] = ((stats.loc[obp_denominator_mask, 'hit_count'] + 
                                                     stats.loc[obp_denominator_mask, 'bb_count']) / 
                                                    stats.loc[obp_denominator_mask, 'obp_denominator']).round(3)
    
    # 長打率
    stats['total_bases'] = (stats['single_count'] + 
                           (2 * stats['double_count']) + 
                           (3 * stats['triple_count']) + 
                           (4 * stats['homerun_count']))
    stats['slugging_pct'] = np.nan
    stats.loc[at_bats_mask, 'slugging_pct'] = (stats.loc[at_bats_mask, 'total_bases'] / 
                                              stats.loc[at_bats_mask, 'at_bats']).round(3)
    
    # OPS
    stats['ops'] = (stats['on_base_pct'].fillna(0) + stats['slugging_pct'].fillna(0)).round(3)
    # NaNの場合は0として計算したOPSをNaNに戻す
    stats.loc[stats['on_base_pct'].isna() | stats['slugging_pct'].isna(), 'ops'] = np.nan
    
    # 三振率
    stats['so_rate'] = np.where(
        stats['plate_appearances'] > 0,
        (stats['strikeout_count'] / stats['plate_appearances'] * 100).round(1),
        0
    )
    
    # 結果のデータフレームを作成
    result_df = pd.DataFrame({
        "打者": stats['Batter'],
        "打席": stats['plate_appearances'],
        "打数": stats['at_bats'],
        "安打": stats['hit_count'],
        "単打": stats['single_count'],
        "二塁打": stats['double_count'],
        "三塁打": stats['triple_count'],
        "本塁打": stats['homerun_count'],
        "打点": stats['RunsScored'],
        "盗塁": stats['stolen_bases'],
        "盗塁死": stats['caught_stealing'],
        "盗塁成功率": stats['sb_success_rate'],
        "犠打": stats['sac_bunt_count'],
        "犠飛": stats['sac_fly_count'],
        "四球": stats['walk_count'],
        "三振": stats['strikeout_count'],
        "打率": stats['batting_avg'],
        "出塁率": stats['on_base_pct'],
        "長打率": stats['slugging_pct'],
        "OPS": stats['ops'],
        "三振率": stats['so_rate']
    })
    
    # 表示列の順序を指定
    columns = ["打者","打席", "打数", "安打", "打率", "打点", "出塁率", "長打率", "OPS", 
                "単打", "二塁打", "三塁打", 
               "本塁打","四球", "三振", 
               "犠打", "犠飛", "三振率", "盗塁", "盗塁死", "盗塁成功率"]
    
    # 存在する列のみを使用して並べ替え
    valid_columns = [col for col in columns if col in result_df.columns]
    result_df = result_df[valid_columns].sort_values("OPS", ascending=False)
    
    return result_df

# 投手成績計算の最適化
@st.cache_data(ttl=86400)
def calculate_stats_pitcher_optimized(df):
    """
    投手成績を計算する関数（超高速化版）- ベクトル化処理とクエリの最小化
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    def calculate_inning_from_float(num):
        """イニング数を文字列表記に変換"""
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
    
    # 最初に必要な列だけを抽出（メモリ使用量削減）
    needed_columns = ['Pitcher', 'PitchCall', 'KorBB', 'PlayResult', 'TaggedHitType', 
                      'OutsOnPlay', 'RunsScored', 'PitchofPA', 'ExitSpeed', 'Angle', 'Direction']
    
    # 存在する列だけを選択
    available_cols = [col for col in needed_columns if col in df.columns]
    df_slim = df[available_cols].copy()
    
    # データフレームに必要な条件フラグを一度に追加（ベクトル化処理）
    ph = ["InPlay", "HitByPitch"]
    kb = ["Strikeout", "Walk"]
    
    # 条件マスクを作成
    if 'PitchCall' in df_slim.columns and 'KorBB' in df_slim.columns:
        df_slim['is_plate_appearance'] = df_slim['PitchCall'].isin(ph) | df_slim['KorBB'].isin(kb)
        df_slim['is_strikeout'] = df_slim['KorBB'] == "Strikeout"
        df_slim['is_walk'] = df_slim['KorBB'] == "Walk"
        df_slim['is_hit_by_pitch'] = df_slim['PitchCall'] == "HitByPitch"
    else:
        df_slim['is_plate_appearance'] = False
        df_slim['is_strikeout'] = False
        df_slim['is_walk'] = False
        df_slim['is_hit_by_pitch'] = False
    
    if 'PlayResult' in df_slim.columns:
        df_slim['is_single'] = df_slim['PlayResult'] == "Single"
        df_slim['is_double'] = df_slim['PlayResult'] == "Double"
        df_slim['is_triple'] = df_slim['PlayResult'] == "Triple"
        df_slim['is_homerun'] = df_slim['PlayResult'] == "HomeRun"
    else:
        df_slim['is_single'] = False
        df_slim['is_double'] = False
        df_slim['is_triple'] = False
        df_slim['is_homerun'] = False
    
    if 'PitchCall' in df_slim.columns:
        df_slim['is_in_play'] = df_slim['PitchCall'] == "InPlay"
        df_slim['is_strike'] = df_slim['PitchCall'] != "BallCalled"
    else:
        df_slim['is_in_play'] = False
        df_slim['is_strike'] = False
    
    # 打球タイプの条件
    if 'TaggedHitType' in df_slim.columns and 'PitchCall' in df_slim.columns:
        df_slim['is_ground_ball'] = (df_slim['TaggedHitType'] == "GroundBall") & (df_slim['PitchCall'] == "InPlay")
        df_slim['is_line_drive'] = (df_slim['TaggedHitType'] == "LineDrive") & (df_slim['PitchCall'] == "InPlay")
        df_slim['is_fly_ball'] = (df_slim['TaggedHitType'] == "FlyBall") & (df_slim['PitchCall'] == "InPlay")
    else:
        df_slim['is_ground_ball'] = False
        df_slim['is_line_drive'] = False
        df_slim['is_fly_ball'] = False
    
    # 打席データのみに絞る
    plate_app_df = df_slim[df_slim['is_plate_appearance']]
    
    # 空のデータフレームの場合は、空の結果を返す
    if plate_app_df.empty:
        return pd.DataFrame()
    
    # 結果を格納するリスト
    result = []
    
    # 各投手のデータを処理
    for pitcher_name, pitcher_df in df_slim.groupby('Pitcher'):
        try:
            # 打席のあるデータだけに絞る
            pa_pitcher_df = pitcher_df[pitcher_df['is_plate_appearance']]
            
            if pa_pitcher_df.empty:
                continue
            
            # アウトカウントと得点の処理
            try:
                outs = pitcher_df['OutsOnPlay'].fillna(0).astype(int).sum() if 'OutsOnPlay' in pitcher_df.columns else 0
                strikeouts = pitcher_df['is_strikeout'].sum()
                total_outs = outs + strikeouts
                ini = total_outs / 3
                ini_str = calculate_inning_from_float(ini)
            except Exception as e:
                st.warning(f"投手 {pitcher_name} のイニング計算中にエラー: {e}")
                ini = 0
                ini_str = "-"
            
            # 投球データの処理
            df_pitches = pitcher_df.dropna(subset=['PitchofPA']) if 'PitchofPA' in pitcher_df.columns else pd.DataFrame()
            total_pitches = len(df_pitches) if not df_pitches.empty else 0
            
            # 三振と四球
            strikeouts = pitcher_df['is_strikeout'].sum()
            walks = pitcher_df['is_walk'].sum()
            
            # 打席数と得点
            plate_appearances = pa_pitcher_df.shape[0]
            runs = pitcher_df['RunsScored'].fillna(0).astype(int).sum() if 'RunsScored' in pitcher_df.columns else 0
            
            # ストライク率
            strikes = df_pitches['is_strike'].sum() if not df_pitches.empty and 'is_strike' in df_pitches.columns else 0
            strike_rate = round(strikes / total_pitches, 2) if total_pitches > 0 else 0
            
            # K-BB%
            k_bb_pct = round((strikeouts - walks) / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            
            # 被安打率
            hits = (pa_pitcher_df['is_single'].sum() + pa_pitcher_df['is_double'].sum() + 
                    pa_pitcher_df['is_triple'].sum() + pa_pitcher_df['is_homerun'].sum())
            
            at_bats = plate_appearances - walks - pa_pitcher_df['is_hit_by_pitch'].sum()
            batting_avg = round(hits / at_bats, 3) if at_bats > 0 else np.nan
            
            # 被OPS計算
            # 出塁率
            on_base = hits + walks + pa_pitcher_df['is_hit_by_pitch'].sum()
            obp = round(on_base / plate_appearances, 3) if plate_appearances > 0 else np.nan
            
            # 長打率
            total_bases = (pa_pitcher_df['is_single'].sum() + 
                          (2 * pa_pitcher_df['is_double'].sum()) + 
                          (3 * pa_pitcher_df['is_triple'].sum()) + 
                          (4 * pa_pitcher_df['is_homerun'].sum()))
            slg = round(total_bases / at_bats, 3) if at_bats > 0 else np.nan
            
            # OPS
            ops = round(obp + slg, 3) if not (np.isnan(obp) or np.isnan(slg)) else np.nan
            
            # K%, BB%, 打球分布
            k_pct = round(strikeouts / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            bb_pct = round(walks / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            
            gb_pct = round(pitcher_df['is_ground_ball'].sum() / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            ld_pct = round(pitcher_df['is_line_drive'].sum() / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            fb_pct = round(pitcher_df['is_fly_ball'].sum() / plate_appearances * 100, 1) if plate_appearances > 0 else 0
            
            # 走者数を計算
            num_runner = (pa_pitcher_df['is_single'].sum() + pa_pitcher_df['is_double'].sum() + 
                          pa_pitcher_df['is_triple'].sum() + pa_pitcher_df['is_homerun'].sum() + 
                          walks + pa_pitcher_df['is_hit_by_pitch'].sum())
            
            # WHIP
            whip = round(num_runner / ini, 2) if ini > 0 else 0
            
            # 結果を配列に追加
            pitcher_stats = [
                pitcher_name, ini_str, plate_appearances, runs, 
                hits, strikeouts, walks, strike_rate, 
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

def display_with_fixed_columns(stats_df, key_column, use_pagination=True, rows_per_page=14, min_column_width=80):
    """
    AgGridを使用して特定の列を固定表示するデータフレーム表示関数
    列幅を調整して表示を改善
    
    Args:
        stats_df: 表示するデータフレーム
        key_column: 固定する列名（例："打者"や"Pitcher"）
        use_pagination: ページネーションを使用するかどうか
        rows_per_page: 1ページあたりの行数
        min_column_width: 列の最小幅（ピクセル）
    """
    # GridOptionsBuilderを使ってオプションを設定
    gb = GridOptionsBuilder.from_dataframe(stats_df)
    
    # key_column列を固定する設定
    gb.configure_column(key_column, pinned="left", minWidth=min_column_width)
    
    # 各列の設定
    for col in stats_df.columns:
        if col != key_column:
            # 列幅を適切に設定
            # 列名の長さや内容に応じて調整することも可能
            gb.configure_column(col, minWidth=min_column_width, autoHeight=True)
    
    # 共通の列設定
    gb.configure_default_column(
        resizable=True,
        filterable=True,
        sortable=True,
        wrapText=True,  # テキストの折り返しを有効化
    )
    
    # ページネーション設定
    if use_pagination:
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=rows_per_page)
    
    # グリッドオプションを取得
    grid_options = gb.build()
    
    # グリッド幅を広げるためのカスタムCSS
    grid_height = 500  # 高さの設定（適宜調整）
    
    # AgGridでデータを表示
    return AgGrid(
        stats_df,
        gridOptions=grid_options,
        height=grid_height,
        width="100%",  # 幅を100%に設定して表示領域を最大化
        fit_columns_on_grid_load=False,  # 自動フィットをオフに
        allow_unsafe_jscode=True,
        theme="streamlit",  # テーマ設定
        update_mode="MODEL_CHANGED",  # より効率的な更新モード
    )

# --- 追加：リザルトプロファイル関数（ベクトル化版） ---
@st.cache_data(ttl=86400)
def calculate_result_profile_pitcher_vectorized(df, pitcher=None):
    '''
    リザルトプロファイルのベクトル化バージョン
    球種の順番を ["FastBall","TwoSeamFastBall","Cutter","Slider","Curveball","Splitter","Changeup","Sinker"]
    に従って並べる。データ存在しない球種はスキップする。
    '''
    if df is None or df.empty:
        return pd.DataFrame()
    
    # 投手データのフィルタリング
    if pitcher is not None:
        df_pitcher = df[df["Pitcher"] == pitcher].copy()
    else:
        df_pitcher = df.copy()
    
    if df_pitcher.empty:
        return pd.DataFrame()
    
    # 指定する球種の順番
    pitch_type_order = ["Fastball", "TwoSeamFastBall", "Cutter", "Slider", 
                        "Curveball", "Splitter", "ChangeUp", "Sinker"]
    
    # 定数設定：ストライク判定リスト
    strikes = ["StrikeSwinging", "StrikeCalled", "FoulBall", "InPlay"]
    
    # 結果を格納するリスト
    result = []
    
    # 有効な球種だけ処理
    if "TaggedPitchType" in df_pitcher.columns:
        valid_pitch_types = [pt for pt in pitch_type_order if pt in df_pitcher["TaggedPitchType"].unique()]
        
        # 各球種と条件ごとに統計を計算
        for pitch_type in valid_pitch_types:
            pitch_df = df_pitcher[df_pitcher["TaggedPitchType"] == pitch_type]
            
            # 全体と2ストライクのケース
            for condition_name, condition_value in [("全体", None), ("2st", 2)]:
                if condition_value is None:
                    df_subset = pitch_df
                else:
                    df_subset = pitch_df[pitch_df["Strikes"] == condition_value]
                
                # 有効なピッチ数
                df_subset_valid = df_subset.dropna(subset=["PitchofPA"])
                tamakazu = len(df_subset_valid)
                
                if tamakazu == 0:
                    rates = [np.nan] * 10  # npを使用して互換性確保
                else:
                    # ベクトル化した計算
                    # 各ピッチコールのベクトル化条件
                    is_swinging = df_subset["PitchCall"] == "StrikeSwinging"
                    is_called = df_subset["PitchCall"] == "StrikeCalled"
                    is_foul = df_subset["PitchCall"] == "FoulBall"
                    is_ball = df_subset["PitchCall"] == "BallCalled"
                    is_inplay = df_subset["PitchCall"] == "InPlay"
                    
                    # 打球タイプ条件
                    is_groundball = is_inplay & (df_subset["TaggedHitType"] == "GroundBall")
                    is_linedrive = is_inplay & (df_subset["TaggedHitType"] == "LineDrive")
                    is_flyball = is_inplay & (df_subset["TaggedHitType"] == "FlyBall")
                    
                    # ストライク条件
                    is_strike = df_subset["PitchCall"].isin(strikes)
                    
                    # 各比率の計算
                    rates = [
                        round(is_swinging.sum() / tamakazu * 100, 1),
                        round(is_called.sum() / tamakazu * 100, 1),
                        round(is_foul.sum() / tamakazu * 100, 1),
                        round(is_ball.sum() / tamakazu * 100, 1),
                        round(is_groundball.sum() / tamakazu * 100, 1),
                        round(is_linedrive.sum() / tamakazu * 100, 1),
                        round(is_flyball.sum() / tamakazu * 100, 1),
                        round(is_strike.sum() / tamakazu * 100, 1)
                    ]
                    
                    # 打球速度関連の統計
                    if "ExitSpeed" in df_subset.columns and not df_subset["ExitSpeed"].dropna().empty:
                        hard_hit = (df_subset["ExitSpeed"] >= 140).sum()
                        hard_hit_rate = round(hard_hit / tamakazu * 100, 1)
                        effective_strike_rate = round(100 - rates[3] - hard_hit_rate, 1)
                        rates.append(hard_hit_rate)
                        rates.append(effective_strike_rate)
                    else:
                        rates.append(np.nan)
                        rates.append(np.nan)
                
                result.append([pitch_type, condition_name] + rates)
        
        # 全体統計の計算（球種によらない）
        for condition_name, condition_value in [("全体", None), ("2st", 2)]:
            if condition_value is None:
                df_total = df_pitcher
            else:
                df_total = df_pitcher[df_pitcher["Strikes"] == condition_value]
            
            # 有効なピッチ数
            df_total_valid = df_total.dropna(subset=["PitchofPA"])
            tamakazu = len(df_total_valid)
            
            if tamakazu == 0:
                rates = [np.nan] * 10
            else:
                # ベクトル化した計算
                is_swinging = df_total["PitchCall"] == "StrikeSwinging"
                is_called = df_total["PitchCall"] == "StrikeCalled"
                is_foul = df_total["PitchCall"] == "FoulBall"
                is_ball = df_total["PitchCall"] == "BallCalled"
                is_inplay = df_total["PitchCall"] == "InPlay"
                
                is_groundball = is_inplay & (df_total["TaggedHitType"] == "GroundBall")
                is_linedrive = is_inplay & (df_total["TaggedHitType"] == "LineDrive")
                is_flyball = is_inplay & (df_total["TaggedHitType"] == "FlyBall")
                
                is_strike = df_total["PitchCall"].isin(strikes)
                
                rates = [
                    round(is_swinging.sum() / tamakazu * 100, 1),
                    round(is_called.sum() / tamakazu * 100, 1),
                    round(is_foul.sum() / tamakazu * 100, 1),
                    round(is_ball.sum() / tamakazu * 100, 1),
                    round(is_groundball.sum() / tamakazu * 100, 1),
                    round(is_linedrive.sum() / tamakazu * 100, 1),
                    round(is_flyball.sum() / tamakazu * 100, 1),
                    round(is_strike.sum() / tamakazu * 100, 1)
                ]
                
                if "ExitSpeed" in df_total.columns and not df_total["ExitSpeed"].dropna().empty:
                    hard_hit = (df_total["ExitSpeed"] >= 140).sum()
                    hard_hit_rate = round(hard_hit / tamakazu * 100, 1)
                    effective_strike_rate = round(100 - rates[3] - hard_hit_rate, 1)
                    rates.append(hard_hit_rate)
                    rates.append(effective_strike_rate)
                else:
                    rates.append(np.nan)
                    rates.append(np.nan)
            
            # 全体行は球種名を "全体" として出力
            result.append(["全体", condition_name] + rates)
    
    # カラム名の設定
    columns = ['球種', '条件', '空振り率', '見逃し率', 'ファウル率', 'ボール率',
               'ゴロ率', 'ライナー率', 'フライ率', 'ストライク率', 'ハードヒット率', '有効ストライク率']
    
    # 結果をデータフレームに変換
    result_df = pd.DataFrame(result, columns=columns)
    
    return result_df

# PyArrow互換の表示用関数
def safe_display_dataframe(df):
    """PyArrow互換の表示用に安全にデータ変換する"""
    if df is None or df.empty:
        return df
    
    # 表示用にコピー
    df_display = df.copy()
    
    # すべての列をチェック
    for col in df_display.columns:
        # NaNを適切に処理
        if df_display[col].dtype.kind in 'fc':  # 浮動小数点と複素数型
            # 数値型であればNaNを'-'に置換してから文字列化
            df_display[col] = df_display[col].fillna('-').astype(str)
        else:
            # その他の型（オブジェクト型含む）は単に文字列化
            df_display[col] = df_display[col].astype(str)
    
    return df_display

# --- 追加：選手別成績関連の関数 ---
@st.cache_data(ttl=86400)
def get_player_list(df, player_type="batter"):
    """
    選手リストを取得する関数
    """
    if df is None or df.empty:
        return []
    
    if player_type == "batter":
        if "Batter" in df.columns:
            players = df["Batter"].dropna().unique().tolist()
        else:
            players = []
    else:  # pitcher
        if "Pitcher" in df.columns:
            players = df["Pitcher"].dropna().unique().tolist()
        else:
            players = []
    
    return sorted(players)

@st.cache_data(ttl=86400)
def get_player_game_dates(df, player_name, player_type="batter"):
    """
    特定の選手が出場した試合の日付リストを取得する関数
    """
    if df is None or df.empty:
        return []
    
    if player_type == "batter":
        player_df = df[df["Batter"] == player_name]
    else:  # pitcher
        player_df = df[df["Pitcher"] == player_name]
    
    if "Date" in player_df.columns:
        dates = player_df["Date"].dt.date.unique().tolist()
        return sorted(dates)
    else:
        return []

# PitchCallやPlayResultの種類別カウントを効率的に計算
@st.cache_data(ttl=86400)
def compute_player_performance_details(df, player_name, player_type="batter"):
    """
    選手の詳細成績を計算
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    if player_type == "batter":
        player_df = df[df["Batter"] == player_name]
    else:  # pitcher
        player_df = df[df["Pitcher"] == player_name]
    
    if player_df.empty:
        return pd.DataFrame()
    
    # 必要な列だけを選択してメモリ使用量を削減
    needed_columns = ['PitchCall', 'KorBB', 'PlayResult', 'TaggedHitType', 'TaggedPitchType',
                       'ExitSpeed', 'Angle', 'RunsScored', 'PlateLocSide', 'PlateLocHeight']
    available_cols = [col for col in needed_columns if col in player_df.columns]
    df_slim = player_df[available_cols].copy()
    
    # 基本的な打撃結果の集計
    result = {}
    
    # PitchCall (ピッチの結果) 集計
    if 'PitchCall' in df_slim.columns:
        pitch_calls = df_slim['PitchCall'].value_counts().to_dict()
        result.update({f"PitchCall_{k}": v for k, v in pitch_calls.items()})
    
    # KorBB (三振/四球) 集計
    if 'KorBB' in df_slim.columns:
        korbb = df_slim['KorBB'].value_counts().to_dict()
        result.update({f"KorBB_{k}": v for k, v in korbb.items()})
    
    # PlayResult (プレー結果) 集計
    if 'PlayResult' in df_slim.columns:
        play_results = df_slim['PlayResult'].value_counts().to_dict()
        result.update({f"PlayResult_{k}": v for k, v in play_results.items()})
    
    # TaggedHitType (打球タイプ) 集計
    if 'TaggedHitType' in df_slim.columns:
        hit_types = df_slim['TaggedHitType'].value_counts().to_dict()
        result.update({f"HitType_{k}": v for k, v in hit_types.items()})
    
    # TaggedPitchType (球種) 集計
    if 'TaggedPitchType' in df_slim.columns:
        pitch_types = df_slim['TaggedPitchType'].value_counts().to_dict()
        result.update({f"PitchType_{k}": v for k, v in pitch_types.items()})
    
    # 打球データの集計
    if all(col in df_slim.columns for col in ['ExitSpeed', 'Angle']):
        batted_balls = df_slim.dropna(subset=['ExitSpeed', 'Angle'])
        if not batted_balls.empty:
            result['Avg_ExitSpeed'] = round(batted_balls['ExitSpeed'].mean(), 1)
            result['Max_ExitSpeed'] = round(batted_balls['ExitSpeed'].max(), 1)
            result['Avg_Angle'] = round(batted_balls['Angle'].mean(), 1)
    
    # 打点集計
    if 'RunsScored' in df_slim.columns:
        result['Total_RBI'] = df_slim['RunsScored'].sum()
    
    # ストライクゾーン情報の集計
    if all(col in df_slim.columns for col in ['PlateLocSide', 'PlateLocHeight']):
        result['Pitches_In_Zone'] = len(df_slim.query('-0.3 <= PlateLocSide <= 0.3 and 0.45 <= PlateLocHeight <= 1.05'))
        result['Pitches_Out_Zone'] = len(df_slim) - result.get('Pitches_In_Zone', 0)
    
    return pd.Series(result)

# --- advanced_metrics関数を追加 ---
@st.cache_data(ttl=86400)
def compute_advanced_metrics(df, player_name, player_type="batter"):
    """
    pasted-2.txtの機能を使って高度な指標を計算する
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    import math
    
    def clock_face_to_degrees(clock_str: str) -> float:
        """ 時計の角度変換関数 """
        if isinstance(clock_str, float) or not clock_str or clock_str == '-':
            return 0.0
        parts = clock_str.split(":")
        hour = float(parts[0]) if len(parts) > 0 else 0.0
        minute = float(parts[1]) if len(parts) > 1 else 0.0
        if hour == 12:
            hour = 0.0
        return (hour + minute / 60.0) * 30.0 % 360.0

    def degrees_to_clock_face(degrees: float) -> str:
        """ 角度を時計表示に戻す関数 """
        total_hours = degrees / 30.0
        hour = int(total_hours)
        minute = round((total_hours - hour) * 60)
        if minute == 60:
            minute = 0
            hour += 1
        hour = hour % 12 if hour % 12 != 0 else 12
        return f"{hour}:{minute:02d}"

    def average_clock_face(series: pd.Series) -> str:
        """ 時計の平均角度を算出し、時計表示に戻す関数 """
        series = series.dropna()
        if series.empty:
            return '-'
        angles = series.apply(clock_face_to_degrees)
        angles_rad = angles.apply(math.radians)
        sin_mean = angles_rad.apply(math.sin).mean()
        cos_mean = angles_rad.apply(math.cos).mean()
        mean_angle_rad = math.atan2(sin_mean, cos_mean)
        return degrees_to_clock_face(math.degrees(mean_angle_rad) % 360)
    
    if player_type == "batter":
        player_df = df[df["Batter"] == player_name]
        
        # 打者の場合はより単純な指標を計算
        if player_df.empty:
            return pd.DataFrame()
        
        # 必要な列だけを選択
        needed_columns = ['PitchCall', 'KorBB', 'PlayResult', 'TaggedHitType', 
                          'ExitSpeed', 'Angle', 'Direction']
        available_cols = [col for col in needed_columns if col in player_df.columns]
        df_slim = player_df[available_cols].copy()
        
        # 基本指標を計算
        result = {}
        
        # 打球データの計算
        if all(col in df_slim.columns for col in ['ExitSpeed']):
            batted_balls = df_slim.query('PitchCall == "InPlay"').dropna(subset=['ExitSpeed'])
            if not batted_balls.empty:
                # 平均打球速度
                result['平均打球速度'] = round(batted_balls['ExitSpeed'].mean(), 1)
                # 最大打球速度
                result['最大打球速度'] = round(batted_balls['ExitSpeed'].max(), 1)
                # ハードヒット率 (95mph = 152.9km/h以上)
                hard_hits = len(batted_balls[batted_balls['ExitSpeed'] >= 152.9])
                result['ハードヒット率'] = round(hard_hits / len(batted_balls) * 100, 1)
        
        # 打球角度の計算
        if all(col in df_slim.columns for col in ['Angle']):
            angled_balls = df_slim.query('PitchCall == "InPlay"').dropna(subset=['Angle'])
            if not angled_balls.empty:
                # 平均打球角度
                result['平均打球角度'] = round(angled_balls['Angle'].mean(), 1)
                # スイートスポット率 (角度8-32度)
                sweet_spots = len(angled_balls[(angled_balls['Angle'] >= 8) & (angled_balls['Angle'] <= 32)])
                result['スイートスポット率'] = round(sweet_spots / len(angled_balls) * 100, 1)
        
        # 飛距離の計算
        if all(col in df_slim.columns for col in ['Direction', 'Angle', 'ExitSpeed']):
            # ここで何か高度な距離計算が必要な場合は実装する
            pass
        
        return pd.Series(result)
    
    else:  # 投手の場合
        player_df = df[df["Pitcher"] == player_name]
        
        if player_df.empty:
            return pd.DataFrame()
        
        # 球種ごとの詳細指標を計算
        pitch_type_order = ["Fastball", "TwoSeamFastBall", "Cutter", "Slider", "Curveball", "Splitter", "Changeup", "Sinker"]
        result = []
        
        # pasted-2.txtの関数実装を参考に簡略化版を作成
        for pitch_type in pitch_type_order:
            if 'TaggedPitchType' not in player_df.columns or pitch_type not in player_df["TaggedPitchType"].unique():
                continue  # 存在しない球種はスキップ
            
            df_1 = player_df[player_df["TaggedPitchType"] == pitch_type]
            df_pitches = df_1.dropna(subset=["PitchofPA"]).reset_index(drop=True)
            df_kekka = df_1.query('PitchCall in ["InPlay", "HitByPitch"] or KorBB in ["Strikeout", "Walk"]')
            
            # 各球種の基本指標計算
            pitch_data = {
                '球種': pitch_type,
                '球数': len(df_1.dropna(subset=["PitchofPA"])),
                'Hit': len(df_kekka.query('PlayResult in ["Single", "Double", "Triple", "HomeRun"]')),
                'HR': len(df_kekka.query('PlayResult == "HomeRun"')),
                "PV": calculate_runvalue1_vectorized_final(df_pitches, 'sum'),
                "PV/100": calculate_runvalue1_vectorized_final(df_pitches, '100')
            }
            
            # 球速
            if 'RelSpeed' in df_pitches.columns and not df_pitches["RelSpeed"].dropna().empty:
                pitch_data['球速'] = round(df_pitches["RelSpeed"].mean(), 1)
            else:
                pitch_data['球速'] = '-'
                
            # 回転数
            if 'SpinRate' in df_pitches.columns and not df_pitches["SpinRate"].dropna().empty:
                pitch_data['回転数'] = round(df_pitches["SpinRate"].mean(), 1)
            else:
                pitch_data['回転数'] = '-'
                
            # 回転軸
            if "Tilt" in df_pitches.columns:
                pitch_data['回転軸'] = average_clock_face(df_pitches["Tilt"])
            else:
                pitch_data['回転軸'] = '-'
                
            # 回転効率
            if 'SpinEfficiency' in df_pitches.columns and not df_pitches["SpinEfficiency"].dropna().empty:
                pitch_data['回転効率'] = round(df_pitches["SpinEfficiency"].mean(), 1)
            else:
                pitch_data['回転効率'] = '-'
                
            # 変化量
            if 'InducedVertBreak' in df_pitches.columns and not df_pitches["InducedVertBreak"].dropna().empty:
                pitch_data['縦変化'] = round(df_pitches["InducedVertBreak"].mean(), 1)
            else:
                pitch_data['縦変化'] = '-'
                
            if 'HorzBreak' in df_pitches.columns and not df_pitches["HorzBreak"].dropna().empty:
                pitch_data['横変化'] = round(df_pitches["HorzBreak"].mean(), 1)
            else:
                pitch_data['横変化'] = '-'

            if 'RelAngle' in df_pitches.columns and not df_pitches["RelAngle"].dropna().empty:
                pitch_data['リリース角'] = round(df_pitches["RelAngle"].mean(), 1)
            else:
                pitch_data['リリース角'] = '-'

            if 'RelHeight' in df_pitches.columns and not df_pitches["RelHeight"].dropna().empty:
                pitch_data['リリース高'] = round(df_pitches["RelHeight"].mean(), 1)
            else:
                pitch_data['リリース高'] = '-'

            if 'RelSide' in df_pitches.columns and not df_pitches["RelSide"].dropna().empty:
                pitch_data['リリース横'] = round(df_pitches["RelSide"].mean(), 1)
            else:
                pitch_data['リリース横'] = '-'

            if 'Extension' in df_pitches.columns and not df_pitches["Extension"].dropna().empty:
                pitch_data['エクステンション'] = round(df_pitches["Extension"].mean(), 1)
            else:
                pitch_data['エクステンション'] = '-'
            
            result.append(pitch_data)
        
        # 全体の集計も追加
        df_total = player_df
        df_total_pitches = df_total.dropna(subset=["PitchofPA"]).reset_index(drop=True)
        df_total_kekka = df_total.query('PitchCall in ["InPlay", "HitByPitch"] or KorBB in ["Strikeout", "Walk"]')

        overall = {
            '球種': "全体",
            '球数': len(df_total.dropna(subset=["PitchofPA"])),
            'Hit': len(df_total_kekka.query('PlayResult in ["Single", "Double", "Triple", "HomeRun"]')),
            'HR': len(df_total_kekka.query('PlayResult == "HomeRun"')),
            "PV": calculate_runvalue1_vectorized_final(df_total_pitches, 'sum'),
            "PV/100": calculate_runvalue1_vectorized_final(df_total_pitches, '100')
        }
        
        # 球速
        if 'RelSpeed' in df_total_pitches.columns and not df_total_pitches["RelSpeed"].dropna().empty:
            overall['球速'] = round(df_total_pitches["RelSpeed"].mean(), 1)
        else:
            overall['球速'] = '-'
            
        # 回転数
        if 'SpinRate' in df_total_pitches.columns and not df_total_pitches["SpinRate"].dropna().empty:
            overall['回転数'] = round(df_total_pitches["SpinRate"].mean(), 1)
        else:
            overall['回転数'] = '-'
            
        # 回転軸
        if "Tilt" in df_total_pitches.columns:
            overall['回転軸'] = average_clock_face(df_total_pitches["Tilt"])
        else:
            overall['回転軸'] = '-'
            
        # 回転効率
        if 'SpinEfficiency' in df_total_pitches.columns and not df_total_pitches["SpinEfficiency"].dropna().empty:
            overall['回転効率'] = round(df_total_pitches["SpinEfficiency"].mean(), 1)
        else:
            overall['回転効率'] = '-'
            
        # 変化量
        if 'InducedVertBreak' in df_total_pitches.columns and not df_total_pitches["InducedVertBreak"].dropna().empty:
            overall['縦変化'] = round(df_total_pitches["InducedVertBreak"].mean(), 1)
        else:
            overall['縦変化'] = '-'
            
        if 'HorzBreak' in df_total_pitches.columns and not df_total_pitches["HorzBreak"].dropna().empty:
            overall['横変化'] = round(df_total_pitches["HorzBreak"].mean(), 1)
        else:
            overall['横変化'] = '-'

        if 'RelAngle' in df_pitches.columns and not df_pitches["RelAngle"].dropna().empty:
            pitch_data['リリース角'] = round(df_pitches["RelAngle"].mean(), 1)
        else:
            pitch_data['リリース角'] = '-'
        if 'RelHeight' in df_pitches.columns and not df_pitches["RelHeight"].dropna().empty:
            pitch_data['リリース高'] = round(df_pitches["RelHeight"].mean(), 1)
        else:
            pitch_data['リリース高'] = '-'
        if 'RelSide' in df_pitches.columns and not df_pitches["RelSide"].dropna().empty:
            pitch_data['リリース横'] = round(df_pitches["RelSide"].mean(), 1)
        else:
            pitch_data['リリース横'] = '-'
        if 'Extension' in df_pitches.columns and not df_pitches["Extension"].dropna().empty:
            pitch_data['エクステンション'] = round(df_pitches["Extension"].mean(), 1)
        else:
            pitch_data['エクステンション'] = '-'
        
        result.append(overall)
        
        return pd.DataFrame(result)

# --- チーム成績ページ（既存機能） ---
def team_stats_page(df):
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
                use_pagination = False
                rows_per_page = 14
                
                # AgGridを使用して列固定表示
                display_with_fixed_columns(stats_df, "打者", use_pagination, rows_per_page)
                
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
                    ['打者','K-BB%', 'K%', 'BB%', 'ストライク率','WHIP', '被打率', '被OPS'],
                    index=0
                )
                ascending = st.checkbox("昇順に並べ替え", value=False)
                stats_df = stats_df.sort_values(sort_column, ascending=ascending)
                
                st.subheader("投手成績")
                use_pagination = False
                rows_per_page = 14
                
                # AgGridを使用して列固定表示
                display_with_fixed_columns(stats_df, "Pitcher", use_pagination, rows_per_page)
                
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

# --- 選手成績ページ（新機能） ---
def player_stats_page(df):
    st.subheader("選手個人成績")
    
    with st.expander("使い方"):
        st.write("1. 分析したい「選手タイプ」（打者 / 投手）を選択してください。")
        st.write("2. 「選手名」を検索ボックスから選択または入力してください。")
        st.write("3. 特定の試合データのみを分析したい場合は、「試合日」から日付を選択してください。")
        st.write("4. 「分析する」ボタンをクリックすると、選手の詳細な成績データが表示されます。")
    
    # 選手タイプ選択
    player_type = st.radio("選手タイプを選択", ["打者", "投手"], horizontal=True)
    
    # GameLevelフィルタ
    st.sidebar.subheader("GameLevel")
    level_filter = []
    if st.sidebar.checkbox("A", value=True, key="player_A"):
        level_filter.append("A")
    if st.sidebar.checkbox("B", value=True, key="player_B"):
        level_filter.append("B")
    
    if not level_filter:
        st.warning("少なくとも一方のGameLevel（A戦またはB戦）を選択してください。")
        return
    
    # フィルタリング（レベルのみ適用）
    filtered_df = df[df["Level"].isin(level_filter)] if level_filter else df
    
    # 選手リスト取得
    player_type_for_list = "batter" if player_type == "打者" else "pitcher"
    player_list = get_player_list(filtered_df, player_type_for_list)
    
    # 選手名検索/選択
    selected_player = st.selectbox(
        "選手名を選択", 
        [""] + player_list,
        format_func=lambda x: x if x else "選手を選択してください"
    )
    
    if selected_player:
        # 選手の出場試合日リスト取得
        game_dates = get_player_game_dates(filtered_df, selected_player, player_type_for_list)
        
        # 日付選択オプション
        st.write(f"{selected_player} の出場試合: {len(game_dates)}試合")
        
        selected_dates = st.multiselect(
            "特定の試合日を選択 (空欄の場合は全試合が対象)",
            options=game_dates,
            format_func=lambda x: x.strftime("%Y-%m-%d") if isinstance(x, date) else str(x)
        )
        
        # 分析実行ボタン
        analyze_button = st.button("分析する")
        
        if analyze_button:
            with st.spinner(f"{selected_player} の成績を分析中..."):
                # 日付でフィルタリング
                player_key = "Batter" if player_type == "打者" else "Pitcher"
                player_data = filtered_df[filtered_df[player_key] == selected_player]
                
                if selected_dates:
                    # 選択された日付でフィルタリング
                    player_data = player_data[player_data["Date"].dt.date.isin(selected_dates)]
                
                if player_data.empty:
                    st.warning("選択された条件に一致するデータがありません。")
                    return
                
                # タブで表示を分ける
                tab1, tab2, tab3 = st.tabs(["基本成績", "詳細指標", "球種/球質分析"])
                
                with tab1:
                    st.subheader(f"{selected_player} - 基本成績")
                    
                    # 試合数と対戦数の表示
                    game_count = len(player_data["Date"].dt.date.unique())
                    
                    if player_type == "打者":
                        # 打者基本成績
                        batter_stats = compute_batter_stats_optimized(player_data)
                        
                        if not batter_stats.empty:
                            # 主要成績をハイライト表示
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("試合数", f"{game_count}試合")
                            with col2:
                                st.metric("打率", f"{batter_stats['打率'].iloc[0]:.3f}" if not pd.isna(batter_stats['打率'].iloc[0]) else "-")
                            with col3:
                                st.metric("OPS", f"{batter_stats['OPS'].iloc[0]:.3f}" if not pd.isna(batter_stats['OPS'].iloc[0]) else "-")
                            with col4:
                                st.metric("本塁打", f"{int(batter_stats['本塁打'].iloc[0])}" if not pd.isna(batter_stats['本塁打'].iloc[0]) else "0")
                            
                            # 詳細成績表示
                            st.write("### 詳細成績")
                            st.dataframe(batter_stats.T)  # 転置して表示
                        else:
                            st.warning("基本成績の計算に必要なデータがありません。")
                    
                    else:  # 投手
                        # 投手基本成績
                        pitcher_stats = calculate_stats_pitcher_optimized(player_data)
                        
                        if not pitcher_stats.empty:
                            # 主要成績をハイライト表示
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("試合数", f"{game_count}試合")
                            with col2:
                                st.metric("投球回", f"{pitcher_stats['回'].iloc[0]}")
                            with col3:
                                st.metric("奪三振", f"{int(pitcher_stats['K'].iloc[0])}" if not pd.isna(pitcher_stats['K'].iloc[0]) else "0")
                            with col4:
                                st.metric("WHIP", f"{pitcher_stats['WHIP'].iloc[0]:.2f}" if not pd.isna(pitcher_stats['WHIP'].iloc[0]) else "-")
                            
                            # 詳細成績表示
                            st.write("### 詳細成績")
                            st.dataframe(pitcher_stats)  # 転置して表示
                        else:
                            st.warning("基本成績の計算に必要なデータがありません。")
                
                with tab2:
                    st.subheader(f"{selected_player} - 詳細指標")
                    
                    # 詳細成績の計算と表示
                    player_details = compute_player_performance_details(player_data, selected_player, player_type_for_list)
                    
                    if not player_details.empty:
                        if player_type == "打者":
                            # 打者詳細指標（選択的に表示）
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("#### 打撃結果")
                                # 安打タイプごとのカウント
                                hit_data = {
                                    "単打": player_details.get("PlayResult_Single", 0),
                                    "二塁打": player_details.get("PlayResult_Double", 0),
                                    "三塁打": player_details.get("PlayResult_Triple", 0),
                                    "本塁打": player_details.get("PlayResult_HomeRun", 0),
                                    "四球": player_details.get("KorBB_Walk", 0),
                                    "三振": player_details.get("KorBB_Strikeout", 0),
                                }
                                st.dataframe(pd.Series(hit_data))
                            
                            with col2:
                                st.write("#### 打球タイプ")
                                # 打球タイプごとのカウント
                                hit_type_data = {
                                    "ゴロ": player_details.get("HitType_GroundBall", 0),
                                    "ライナー": player_details.get("HitType_LineDrive", 0),
                                    "フライ": player_details.get("HitType_FlyBall", 0),
                                    "ポップ": player_details.get("HitType_PopUp", 0),
                                    "バント": player_details.get("HitType_Bunt", 0),
                                }
                                st.dataframe(pd.Series(hit_type_data))
                            
                            # 打球データ
                            st.write("#### 打球データ")
                            batted_ball_data = {
                                "平均打球速度": player_details.get("Avg_ExitSpeed", "-"),
                                "最大打球速度": player_details.get("Max_ExitSpeed", "-"),
                                "平均打球角度": player_details.get("Avg_Angle", "-"),
                            }
                            st.dataframe(pd.Series(batted_ball_data))
                            
                        else:  # 投手
                            # 投手詳細指標
                            st.write("#### 球種別詳細指標")
                            data = calculate_result_profile_pitcher_vectorized(player_data,pitcher = selected_player)
                            st.dataframe(data)
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("#### 投球結果")
                                # 投球結果ごとのカウント
                                pitch_data = {
                                    "被安打": player_details.get("PlayResult_Single", 0) + player_details.get("PlayResult_Double", 0) + 
                                            player_details.get("PlayResult_Triple", 0) + player_details.get("PlayResult_HomeRun", 0),
                                    "被本塁打": player_details.get("PlayResult_HomeRun", 0),
                                    "奪三振": player_details.get("KorBB_Strikeout", 0),
                                    "四球": player_details.get("KorBB_Walk", 0),
                                    "空振り": player_details.get("PitchCall_StrikeSwinging", 0),
                                    "見逃し": player_details.get("PitchCall_StrikeCalled", 0),
                                }
                                st.dataframe(pd.Series(pitch_data))
                            
                            with col2:
                                st.write("#### 被打球タイプ")
                                # 被打球タイプごとのカウント
                                hit_type_data = {
                                    "ゴロ": player_details.get("HitType_GroundBall", 0),
                                    "ライナー": player_details.get("HitType_LineDrive", 0),
                                    "フライ": player_details.get("HitType_FlyBall", 0),
                                    "ポップ": player_details.get("HitType_PopUp", 0),
                                }
                                st.dataframe(pd.Series(hit_type_data))
                    else:
                        st.warning("詳細指標の計算に必要なデータがありません。")
                
                with tab3:
                    st.subheader(f"{selected_player} - 球種/球質分析")
                    
                    # 高度な指標の計算と表示
                    advanced_metrics = compute_advanced_metrics(player_data, selected_player, player_type_for_list)
                    
                    if player_type == "打者":
                        if isinstance(advanced_metrics, pd.Series) and not advanced_metrics.empty:
                            st.write("#### 打球品質指標")
                            st.dataframe(advanced_metrics)
                        else:
                            st.warning("球質分析に必要なデータがありません。")
                    else:  # 投手
                        if isinstance(advanced_metrics, pd.DataFrame) and not advanced_metrics.empty:
                            st.write("#### 球種別指標")
                            st.dataframe(advanced_metrics)
                            
                            # 主要球種のハイライト表示
                            # 全体行を除く
                            pitch_types = advanced_metrics[advanced_metrics['球種'] != '全体']
                            
                            if not pitch_types.empty:
                                st.write("#### 主要球種")
                                
                                # 使用率トップ3の球種を表示
                                top_pitches = pitch_types.sort_values('球数', ascending=False).head(3)
                                
                                col1, col2, col3 = st.columns(3)
                                for i, (idx, row) in enumerate(top_pitches.iterrows()):
                                    with [col1, col2, col3][i]:
                                        pitch_name = row['球種']
                                        usage_pct = round(row['球数'] / advanced_metrics[advanced_metrics['球種'] == '全体']['球数'].iloc[0] * 100, 1)
                                        velo = row['球速'] if row['球速'] != '-' else '--'
                                        
                                        st.write(f"**{pitch_name}**")
                                        st.write(f"使用率: {usage_pct}%")
                                        st.write(f"球速: {velo}")
                                        
                                        # 変化量が有効な値であれば表示
                                        if row['縦変化'] != '-' and row['横変化'] != '-':
                                            st.write(f"変化量: 縦{row['縦変化']}cm / 横{row['横変化']}cm")
                        else:
                            st.warning("球種分析に必要なデータがありません。")
    else:
        st.info("選手を選択して分析を開始してください。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"アプリケーション実行中にエラーが発生しました: {e}")
        st.info("詳細なエラー情報を確認するには、Streamlitのデバッグモードを有効にしてください。")
