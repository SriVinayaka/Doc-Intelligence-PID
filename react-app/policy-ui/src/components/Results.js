


import React from 'react'
import { useSelector } from 'react-redux'
import PolicyCard from './PolicyCard'

const Results = () => {
  const { results, loading, error } = useSelector((state) => state.search)

  if (loading) return <p>Loading...</p>
  if (error) return <p>Error: {error}</p>
  if (!results) return null

  return (
    <div>

      {/* Exact Policy */}
      {results.exact_policy && (
        <>
          <h3>✅ Exact Policy</h3>
          <PolicyCard data={results.exact_policy} />
        </>
      )}

      {/* Field Matches */}
      {results.field_matches?.length > 0 && (
        <>
          <h3>🎯 Field Matches</h3>
          {results.field_matches.map((item, idx) => (
            <PolicyCard key={idx} data={item} />
          ))}
        </>
      )}

      {/* Semantic Matches */}
      {results.semantic_matches?.length > 0 && (
        <>
          <h3>🧠 Semantic Matches</h3>
          {results.semantic_matches.map((item, idx) => (
            <PolicyCard key={idx} data={item} />
          ))}
        </>
      )}

      {/* No Results */}
      {(!results.field_matches?.length &&
        !results.semantic_matches?.length &&
        !results.exact_policy) && (
        <p>No results found</p>
      )}

    </div>
  )
}

export default Results

