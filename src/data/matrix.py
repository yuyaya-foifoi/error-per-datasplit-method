from scipy import sparse


def to_matrix(data, shape=(3883, 6040)):
    ratings = data.Rating
    movie_id = data.MovieID
    user_id = data.UserID
    return sparse.csr_matrix(
        (ratings, (movie_id, user_id)), shape=(shape[0], shape[1])
    ).toarray()
