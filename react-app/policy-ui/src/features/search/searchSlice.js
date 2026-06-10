import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import { fetchSearchResults } from './searchAPI'

export const searchQuery = createAsyncThunk(
  'search/query',
  async (payload) => {
    return await fetchSearchResults(payload)
  }
)

const searchSlice = createSlice({
  name: 'search',
  initialState: {
    results: null,
    loading: false,
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(searchQuery.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(searchQuery.fulfilled, (state, action) => {
        state.loading = false
        state.results = action.payload
      })
      .addCase(searchQuery.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message
      })
  },
})

export default searchSlice.reducer
