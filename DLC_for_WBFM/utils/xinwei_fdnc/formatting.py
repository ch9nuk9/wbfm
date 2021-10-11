from sklearn.preprocessing import StandardScaler


def zimmer2leifer(vol0_zxy):

    scaler = StandardScaler()
    vol0_scaled = scaler.fit_transform(vol0_zxy)
    # Reduce z
    vol0_scaled[:, 0] /= 3.0
    # Reorder dimensions
    vol0_scaled = vol0_scaled[:, [2, 1, 0]]
    # Somehow their point clouds are much smaller than mine
    vol0_scaled /= 5.0

    return vol0_scaled, scaler


def leifer2zimmer(vol0_scaled, scaler):

    # Somehow their point clouds are much smaller than mine
    vol0_scaled *= 5.0
    # Reorder dimensions; coincidentally symmetric
    vol0_scaled = vol0_scaled[:, [2, 1, 0]]
    # Increase z
    vol0_scaled[:, 0] *= 3.0

    vol0_zxy = scaler.inverse_transform(vol0_scaled)

    return vol0_zxy
