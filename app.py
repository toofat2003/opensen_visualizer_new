import streamlit as st
import pandas as pd
import glob
import time
import base64
from datetime import datetime, date
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder

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
               "犠打", "犠飛", "三振率", "盗塁", "盗塁死", "盗塁成功率", ]
    
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"アプリケーション実行中にエラーが発生しました: {e}")
        st.info("詳細なエラー情報を確認するには、Streamlitのデバッグモードを有効にしてください。")