def dataframe_to_standard_zxy_format(df_tracklets):
    coords = ['z', 'x', 'y']
    df_tracklets = df_tracklets.loc(axis=1)[:, coords]
    df_tracklets.sort_index(axis=1, level=0, sort_remaining=False)
    return df_tracklets