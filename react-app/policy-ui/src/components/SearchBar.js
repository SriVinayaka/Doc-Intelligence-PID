

import React, { useState } from 'react'
import { useDispatch } from 'react-redux'
import { searchQuery } from '../features/search/searchSlice'

const SearchBar = () => {
  const dispatch = useDispatch()

  const [query, setQuery] = useState('')
  const [field, setField] = useState('')
  const [value, setValue] = useState('')

  const handleSearch = () => {
    dispatch(
      searchQuery({
        query,
        field,
        value,
        top_k: 5,
      })
    )
  }

  return (
    <div style={{ marginBottom: '20px' }}>
      <input
        placeholder="Query"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <input
        placeholder="Field"
        value={field}
        onChange={(e) => setField(e.target.value)}
      />
      <input
        placeholder="Value"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <button onClick={handleSearch}>Search</button>
    </div>
  )
}

export default SearchBar

